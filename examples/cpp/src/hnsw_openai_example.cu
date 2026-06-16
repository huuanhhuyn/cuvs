/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/memory_tracking_resources.hpp>
#include <string>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/util/host_memory.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

#include <cstdio>
#include <cstdlib>  // for exit
#include <fcntl.h>
#include <optional>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

enum class build_mode { no_ace, ace, ace_use_disk };

int cagra_build_search_ace(raft::resources const& res, build_mode mode)
{
  using namespace cuvs::neighbors;

  // Open dataset in big-ann-benchmarks binary format.
  int fd = open("openai_5M/base.5M.fbin", O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }
  uint32_t shape[2];
  ssize_t bytesRead = read(fd, shape, 8);
  if (bytesRead != 8) {
    perror("Error reading shape");
    close(fd);
    return EXIT_FAILURE;
  }
  size_t data_size = shape[0] * static_cast<size_t>(shape[1]);
  std::cout << "Dataset size " << data_size << std::endl;
  size_t header_size   = sizeof(shape);
  size_t file_size     = data_size * sizeof(float) + header_size;
  uint8_t* dataset_ptr = (uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  std::cout << "shape [" << shape[0] << ", " << shape[1] << "]" << std::endl;
  if (dataset_ptr == MAP_FAILED) {
    perror("Error mmapping the file");
    close(fd);
    return EXIT_FAILURE;
  }
  uint32_t n_rows        = shape[0];
  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    reinterpret_cast<float*>(dataset_ptr + header_size), n_rows, shape[1]);

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

  auto hnsw_index = hnsw::build(res, params, dataset_host_view);

  std::string hnsw_index_path = "hnsw_index.bin";
  cuvs::neighbors::hnsw::serialize(res, hnsw_index_path, *hnsw_index);
  std::cout << "HNSW index file location: " << hnsw_index_path << std::endl;

  munmap(dataset_ptr, file_size);
  close(fd);
  return 0;
}

int main(int argc, char** argv)
{
  // Select graph build mode: "no_ace" (default), "ace", or "ace_use_disk".
  build_mode mode = build_mode::no_ace;
  if (argc > 1) {
    std::string arg = argv[1];
    if (arg == "no_ace") {
      mode = build_mode::no_ace;
    } else if (arg == "ace") {
      mode = build_mode::ace;
    } else if (arg == "ace_use_disk") {
      mode = build_mode::ace_use_disk;
    } else {
      std::cerr << "Usage: " << argv[0] << " [no_ace|ace|ace_use_disk]" << std::endl;
      return EXIT_FAILURE;
    }
  }

  raft::resources res;

  // // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  // rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
  //   rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  // rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  raft::resource::set_workspace_to_pool_resource(res, 2 * 1024 * 1024 * 1024ull);

  const char* mode_str = (mode == build_mode::ace_use_disk) ? "ace_use_disk"
                         : (mode == build_mode::ace)        ? "ace"
                                                            : "no_ace";
  char csv_path_buf[256];
  snprintf(csv_path_buf, sizeof(csv_path_buf), "openai_5M_%s.csv", mode_str);
  const char* csv_path = csv_path_buf;
  raft::memory_tracking_resources tracked(res, csv_path, std::chrono::milliseconds(1));

  // ACE build and search example.
  cagra_build_search_ace(tracked, mode);

  std::cout << "Tracking stats: " << csv_path << std::endl;
}
