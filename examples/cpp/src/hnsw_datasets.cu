/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/memory_tracking_resources.hpp>
#include <string>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/util/host_memory.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdio>
#include <cstdlib>  // for exit
#include <fcntl.h>
#include <optional>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

template <typename T>
class BinaryFile {
 public:
  BinaryFile(const char* filepath, uint32_t max_rows = 0)
    : fd_(-1), mapped_ptr_(nullptr), data_(nullptr), file_size_(0)
  {
    fd_ = open(filepath, O_RDONLY);
    if (fd_ == -1) { throw std::runtime_error(std::string("Error opening file: ") + filepath); }

    uint32_t shape[2];
    ssize_t bytesRead = read(fd_, shape, 8);
    if (bytesRead != 8) {
      close(fd_);
      throw std::runtime_error(std::string("Error reading shape from file: ") + filepath);
    }

    shape_[0] = (max_rows > 0 && max_rows < shape[0]) ? max_rows : shape[0];
    shape_[1] = shape[1];

    size_t data_size   = shape_[0] * static_cast<size_t>(shape_[1]);
    size_t header_size = 8;
    file_size_         = data_size * sizeof(T) + header_size;

    mapped_ptr_ = (uint8_t*)mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_ptr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error(std::string("Error mmapping file: ") + filepath);
    }

    data_ = reinterpret_cast<T*>(mapped_ptr_ + header_size);
  }

  ~BinaryFile()
  {
    if (mapped_ptr_ != nullptr && mapped_ptr_ != MAP_FAILED) { munmap(mapped_ptr_, file_size_); }
    if (fd_ != -1) { close(fd_); }
  }

  BinaryFile(const BinaryFile&)            = delete;
  BinaryFile& operator=(const BinaryFile&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }

  uint32_t rows() const { return shape_[0]; }
  uint32_t cols() const { return shape_[1]; }
  raft::host_matrix_view<const T, int64_t> view()
  { return raft::make_host_matrix_view<const T, int64_t>(data_, rows(), cols()); }

 private:
  int fd_;
  uint8_t* mapped_ptr_;
  T* data_;
  size_t file_size_;
  uint32_t shape_[2];
};

enum class build_mode { no_ace, ace, ace_use_disk };

int main(int argc, char* argv[])
{
  using namespace cuvs::neighbors;

  const char* index_save_path = nullptr;
  uint32_t max_dataset_rows   = 0;
  build_mode mode             = build_mode::no_ace;
  std::vector<const char*> positional_args;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      index_save_path = argv[++i];
    } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      max_dataset_rows = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "no_ace") {
        mode = build_mode::no_ace;
      } else if (m == "ace") {
        mode = build_mode::ace;
      } else if (m == "ace_use_disk") {
        mode = build_mode::ace_use_disk;
      } else {
        std::cerr << "Invalid -m value: " << m << " (expected no_ace|ace|ace_use_disk)"
                  << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      positional_args.push_back(argv[i]);
    }
  }

  if (positional_args.size() != 1) {
    std::cerr << "Usage: " << argv[0]
              << "  [-o index_file] [-n max_rows] [-m no_ace|ace|ace_use_disk] <dataset_file>"
              << std::endl;
    return EXIT_FAILURE;
  }

  raft::resources res_untracked;
  const char* mode_str = (mode == build_mode::ace_use_disk) ? "ace_use_disk"
                         : (mode == build_mode::ace)        ? "ace"
                                                            : "no_ace";
  // Derive a dataset name from the input path, e.g. "/datasets/gist_1M/base.fbin" -> "gist_1M_base".
  std::filesystem::path dataset_path(positional_args[0]);
  std::string dataset_name =
    dataset_path.parent_path().filename().string() + "_" + dataset_path.stem().string();
  std::string csv_path_str = "datasets_" + dataset_name + "_" + mode_str + ".csv";
  const char* csv_path     = csv_path_str.c_str();

  // Define a pool allocator for temporary arrays. Internal arrays would use the pool, any other
  // allocation uses the default RMM memory resource. We set a pool with 2 GiB upper limit.
  raft::resource::set_workspace_to_pool_resource(res_untracked, 2 * 1024 * 1024 * 1024ull);

  raft::memory_tracking_resources res(res_untracked, csv_path, std::chrono::milliseconds(1));

  BinaryFile<float> dataset(positional_args[0], max_dataset_rows);

  std::cout << "Dataset shape: [" << dataset.rows() << ", " << dataset.cols() << "]" << std::endl;

  raft::default_logger().set_level(rapids_logger::level_enum::debug);

  auto start_time = std::chrono::high_resolution_clock::now();

  // HNSW index parameters
  hnsw::index_params params;
  params.M               = 24;
  params.ef_construction = 200;
  params.hierarchy       = cuvs::neighbors::hnsw::HnswHierarchy::GPU;

  if (mode == build_mode::no_ace) {
    std::cout << "Using non-ACE graph build" << std::endl;
  } else {
    auto ace_params           = hnsw::graph_build_params::ace_params();
    ace_params.npartitions    = 4;
    ace_params.build_dir      = "/tmp/hnsw_ace_build";
    ace_params.use_disk       = (mode == build_mode::ace_use_disk);
    params.graph_build_params = ace_params;
    std::cout << "Using ACE graph build (use_disk=" << std::boolalpha << ace_params.use_disk << ")"
              << std::endl;
  }

  std::cout << "Building HNSW index" << std::endl;
  auto hnsw_index = hnsw::build(res, params, dataset.view());

  std::cout << "Serializing HNSW index to " << index_save_path << std::endl;
  cuvs::neighbors::hnsw::serialize(res, index_save_path, *hnsw_index);
  std::cout << "HNSW index file location: " << index_save_path << std::endl;

  // auto index_params =
  //   cagra::index_params::from_hnsw_params(dataset.view().extents(),
  //                                         M,
  //                                         params.ef_construction,
  //                                         cagra::hnsw_heuristic_type::SAME_GRAPH_FOOTPRINT,
  //                                         params.metric);

  // std::cout << "Building CAGRA index (search graph)" << std::endl;
  // auto cagra_index = cagra::build(res, index_params, dataset.view());

  // // Convert CAGRA index to HNSW
  // std::cout << "Converting CAGRA index to HNSW" << std::endl;
  // auto hnsw_index = hnsw::from_cagra(res, params, cagra_index, dataset.view());

  // cuvs::neighbors::hnsw::serialize(res, index_save_path, *hnsw_index);
  // std::cout << "HNSW index file location: " << index_save_path << std::endl;

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start_time);
  double avg_time_seconds = duration.count() / 1000.0;
  std::cout << "HNSW index created in in " << avg_time_seconds << " seconds" << std::endl;
  std::cout << "Tracking stats: " << csv_path << std::endl;
}