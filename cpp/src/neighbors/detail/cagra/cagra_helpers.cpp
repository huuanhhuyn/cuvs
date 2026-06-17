/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <iostream>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <utility>

namespace cuvs::neighbors::cagra::helpers {

constexpr double to_mib(size_t bytes) { return static_cast<double>(bytes) / (1 << 20); }
constexpr double to_gib(size_t bytes) { return static_cast<double>(bytes) / (1 << 30); }

// Calculate CAGRA optimize workspace memory requirements.
// This is the working memory on top of the input/output memory usage.
std::tuple<size_t, size_t, size_t, size_t> optimize_workspace_size(size_t n_rows,
                                                                   size_t graph_degree,
                                                                   size_t intermediate_degree,
                                                                   size_t index_size,
                                                                   bool mst_optimize)
{
  RAFT_EXPECTS(graph_degree > 0, "graph_degree must be greater than 0");
  RAFT_EXPECTS(intermediate_degree >= graph_degree,
               "intermediate_degree must be greater than or equal to graph_degree");

  // compare new method
  memuse_optimize(n_rows, graph_degree, intermediate_degree, index_size, mst_optimize);

  // MST optimization memory (host only)
  size_t mst_host       = 0;
  size_t mst_host_fixed = 0;
  if (mst_optimize) {
    mst_host = n_rows * index_size;                  // mst_graph_num_edges
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in optimize
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in mst_optimize
    mst_host += n_rows * index_size * 7;             // vectors with _max_edges suffix
    mst_host_fixed += (graph_degree - 1) * (graph_degree - 1) * index_size;  // iB_candidates
    mst_host += mst_host_fixed;
  }

  // batchsize for both prune and combine stages
  size_t batch_size = std::min(static_cast<size_t>(256 * 1024), n_rows);

  // Prune stage memory
  // We neglect 8 bytes (both on host and device) for stats
  size_t prune_dev_fixed = batch_size * intermediate_degree * 1;  // detour count (uint8_t)
  // CHECK ME: d_num_detour_edges is not used anywhere
  // prune_dev_fixed += batch_size * sizeof(uint32_t);               // d_num_detour_edges
  prune_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)

  size_t prune_dev = n_rows * intermediate_degree * index_size;  // d_input_graph
  prune_dev += prune_dev_fixed;

  // Reverse graph stage memory
  size_t rev_dev = n_rows * graph_degree * index_size;  // d_rev_graph
  rev_dev += n_rows * sizeof(uint32_t);                 // d_rev_graph_count
  rev_dev += n_rows * index_size;                       // d_dest_nodes

  // Memory for merging graphs (host only optional)
  size_t combine_host_fixed = graph_degree * sizeof(uint32_t);  // histogram
  size_t combine_host       = n_rows * sizeof(uint32_t);        // n_edge_count
  combine_host += combine_host_fixed;

  // additional memory for combine stage on device (3 batches)
  size_t combine_dev_fixed = 2 * batch_size * graph_degree * index_size;  // d_output_graph(2*batch)
  if (mst_optimize) {
    combine_dev_fixed += 2 * batch_size * graph_degree * index_size;  // d_mst_graph(2*batch)
    combine_dev_fixed += 2 * batch_size * sizeof(uint32_t);  // d_mst_graph_num_edges(2*batch)
  }
  size_t combine_dev = combine_dev_fixed;

  size_t total_host       = mst_host + combine_host;
  size_t total_host_fixed = mst_host_fixed + combine_host_fixed;
  size_t total_dev        = std::max(prune_dev, rev_dev + combine_dev);
  size_t total_dev_fixed  = std::max(prune_dev_fixed, combine_dev_fixed);

  return std::make_tuple(total_host, total_dev, total_host_fixed, total_dev_fixed);
}

// All sizes are in bytes
inline std::pair<size_t, size_t> ivf_pq_build_mem_usage(
  raft::resources const& res,
  raft::matrix_extent<int64_t> dataset,
  cuvs::neighbors::graph_build_params::ivf_pq_params params,
  size_t graph_degree,
  size_t intermediate_graph_degree)
{
  size_t n_rows = dataset.extent(0);

  size_t dataset_gpu_mem =
    cuvs::neighbors::ivf_pq::helpers::compressed_dataset_size(res, dataset, params.build_params);
  size_t graph_host_mem = n_rows * (graph_degree + intermediate_graph_degree) * sizeof(uint32_t);
  auto [host_workspace_size,
        gpu_workspace_size,
        host_workspace_size_fixed,
        gpu_workspace_size_fixed] =
    cuvs::neighbors::cagra::helpers::optimize_workspace_size(
      n_rows, graph_degree, intermediate_graph_degree, sizeof(uint32_t));

  // The kmeans trainset is a large temporary float buffer allocated during IVF-PQ training.
  // It is freed before the extend phase, so peak GPU = max(training_peak, extend_peak).
  size_t kmeans_trainset_ratio = std::max<size_t>(
    1,
    n_rows / std::max<size_t>(params.build_params.kmeans_trainset_fraction * n_rows,
                              params.build_params.n_lists));
  size_t kmeans_n_rows  = n_rows / kmeans_trainset_ratio;
  size_t kmeans_gpu_mem = kmeans_n_rows * dataset.extent(1) * sizeof(float);

  size_t total_host =
    graph_host_mem + host_workspace_size + 2e9;  // added 2 GB extra workspace (IVF-PQ search)
  size_t total_dev = std::max({kmeans_gpu_mem, dataset_gpu_mem, gpu_workspace_size}) + 1e9;
  return std::make_pair(total_host, total_dev);
}

std::pair<size_t, size_t> cagra_build_mem_usage(raft::resources const& res,
                                                raft::matrix_extent<int64_t> dataset,
                                                size_t dtype_size,
                                                cuvs::neighbors::cagra::index_params cparams)
{
  using namespace cuvs::neighbors;

  size_t total_host = 0;
  size_t total_dev  = 0;

  if (std::holds_alternative<graph_build_params::ivf_pq_params>(cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with IVF-PQ");
    graph_build_params::ivf_pq_params pq_params =
      std::get<graph_build_params::ivf_pq_params>(cparams.graph_build_params);
    std::tie(total_host, total_dev) = ivf_pq_build_mem_usage(
      res, dataset, pq_params, cparams.graph_degree, cparams.intermediate_graph_degree);
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with NN-descent");
    // Build needs dataset in fp16 on dev and graph on dev?
    // dataset copied to device for sorting
    // TODO(tfeher) proper estimate
    total_host = dataset.extent(0) * dataset.extent(1) * dtype_size +
                 dataset.extent(0) * (cparams.graph_degree + cparams.intermediate_graph_degree) *
                   sizeof(uint32_t) +
                 2e9;  // Extra buffer
    total_dev = total_host;
  } else {
    // iterative build
    // TODO(tfeher): proper estimate
    total_host = dataset.extent(0) * dataset.extent(1) * dtype_size +
                 dataset.extent(0) * (cparams.graph_degree + cparams.intermediate_graph_degree) *
                   sizeof(uint32_t) +
                 2e9;  // Extra buffer
    total_dev = total_host;
  }
  return std::make_pair(total_host, total_dev);
}

// Estimate peak memory allocated per source during graph_core::optimize().
MemUsage memuse_optimize(size_t n_rows,
                         size_t graph_degree,
                         size_t intermediate_degree,
                         size_t index_size,
                         bool guarantee_connectivity)
{
  size_t batch_size = std::min(static_cast<size_t>(256 * 1024), n_rows);

  // Peak host: outer + inner mst arrays all live simultaneously (if mst)
  size_t total_host = 0;
  if (guarantee_connectivity) {
    total_host += n_rows * graph_degree * index_size;  // mst_graph (outer, in optimize())
    std::cout << "optimize::mst_graph " << to_mib(n_rows * graph_degree * index_size) << " MiB"
              << std::endl;
    total_host += n_rows * sizeof(uint32_t);  // mst_graph_num_edges (outer)
    std::cout << "optimize::mst_graph_num_edges " << to_mib(n_rows * sizeof(uint32_t)) << " MiB"
              << std::endl;
    total_host += n_rows * graph_degree * index_size;  // mst_graph (inner, in mst_optimization())
    std::cout << "optimize::mst_optimization::mst_graph "
              << to_mib(n_rows * graph_degree * index_size) << " MiB" << std::endl;
    total_host +=
      7 * n_rows *
      index_size;  // outgoing/incoming_max/num_edges, label, cluster_size, candidate_edges
    std::cout << "optimize::mst_optimization::work_vectors " << to_mib(7 * n_rows * index_size)
              << " MiB" << std::endl;
    std::cout << "optimize::mst_optimization::d_mst_graph "
              << to_mib((graph_degree + 1) * n_rows * index_size) << " MiB" << std::endl;
    std::cout << "optimize::mst_optimization::d_work_vectors " << to_mib(7 * n_rows * index_size)
              << " MiB" << std::endl;
  }

  // Peak large_workspace: prune loads full knn graph; d_rev_graph allocated after prune returns
  size_t total_large = n_rows * intermediate_degree * index_size;  // d_input_graph

  // Peak workspace: merge stage (d_rev_graph_count persists + batch iterators)
  size_t total_ws = n_rows * sizeof(uint32_t);             // d_rev_graph_count
  total_ws += 2 * batch_size * graph_degree * index_size;  // d_output_graph (2 batches)
  if (guarantee_connectivity) {
    total_ws += 2 * batch_size * graph_degree * index_size;  // d_mst_graph (2 batches)
    total_ws += 2 * batch_size * sizeof(uint32_t);           // d_mst_graph_num_edges (2 batches)
  }

  // Peak device (default allocator):
  //   mst case: d_mst_graph + 8 vectors in mst_optimization (freed before prune)
  //   no-mst case: d_dest_nodes in make_reverse_graph_gpu (host-accessible output path)
  size_t total_dev = 0;
  if (guarantee_connectivity) {
    total_dev += n_rows * graph_degree * index_size;  // d_mst_graph
    total_dev += 8 * n_rows * index_size;             // d_mst_graph_num_edges + 7 work vectors
  } else {
    total_dev += n_rows * index_size;  // d_dest_nodes
    std::cout << "optimize::make_reverse_graph_gpu::d_dest_nodes " << to_mib(n_rows * index_size)
              << " MiB" << std::endl;
    std::cout << "optimize::make_reverse_graph_gpu::dest_nodes " << to_mib(n_rows * index_size)
              << " MiB" << std::endl;
  }

  std::cout << "optimize::prune::d_input_graph "
            << to_mib(n_rows * intermediate_degree * index_size) << " MiB" << std::endl;
  std::cout << "optimize::prune::d_output_graph "
            << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
  std::cout << "optimize::d_rev_graph " << to_mib(n_rows * graph_degree * index_size) << " MiB"
            << std::endl;
  std::cout << "optimize::d_rev_graph_count " << to_mib(n_rows * sizeof(uint32_t)) << " MiB"
            << std::endl;
  std::cout << "optimize::merge::d_output_graph "
            << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
  if (guarantee_connectivity) {
    std::cout << "optimize::merge::d_mst_graph "
              << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
    std::cout << "optimize::merge::d_mst_graph_num_edges "
              << to_mib(2 * batch_size * sizeof(uint32_t)) << " MiB" << std::endl;
  }
  std::cout << "memuse_optimize: host: " << to_gib(total_host) << " GiB, "
            << "pinned: " << to_gib(0) << " GiB, "
            << "workspace: " << to_gib(total_ws) << " GiB, "
            << "large_workspace: " << to_gib(total_large) << " GiB, "
            << "managed: " << to_gib(0) << " GiB, "
            << "device: " << to_gib(total_dev) << " GiB" << std::endl;

  return {.host            = total_host,
          .pinned          = 0,
          .workspace       = total_ws,
          .large_workspace = total_large,
          .managed         = 0,
          .device          = total_dev};
}

// Estimate peak memory allocated per source during ivf_pq_build.cuh::extend().
MemUsage memuse_ivf_pq_extend(size_t n_rows,
                              size_t dim,
                              size_t rot_dim,
                              size_t n_clusters,
                              size_t index_size,
                              size_t dtype_size,
                              size_t pq_dim,
                              size_t pq_bits)
{
  size_t batch_size = std::min(n_rows, size_t(65536));

  // new_data_labels is allocated via raft large-workspace resource (not the batch workspace).
  size_t new_data_labels_large = n_rows * sizeof(uint32_t);

  size_t label_ws = n_clusters * dim * sizeof(float)  // cluster_centers
                    + batch_size * dim * dtype_size;  // vec_batches_buf (1 buffer, no prefetch)

  // flat_compute_residuals_tmp uses batches_mr (workspace), not the default device allocator.
  size_t fill_ws = batch_size * dim * dtype_size         // vec_batches_buf
                   + batch_size * rot_dim * sizeof(float)  // new_vectors_residual
                   + batch_size * dim * sizeof(float);     // flat_compute_residuals_tmp

  size_t total_ws = std::max(label_ws, fill_ws);

  // interleaved bytes-per-vector for codes = round_up(pq_dim * pq_bits, 128) / 8
  // (each group of kIndexGroupVecLen=16 bytes holds 16*8/pq_bits codes per subspace chunk)
  size_t code_bytes_per_vec = ((pq_dim * pq_bits + 127) / 128) * 16;
  size_t bytes_per_row      = code_bytes_per_vec + index_size;
  // placeholder: one ivf::list of n_rows + 31*n_clusters rows (OOM probe, freed early).
  // ivf::list rounds capacity up to next multiple of align_max=1024.
  size_t placeholder_rows = n_rows + 31 * n_clusters;
  size_t placeholder_cap  = ((placeholder_rows + 1023) / 1024) * 1024;
  size_t placeholder_dev  = placeholder_cap * bytes_per_row;
  // resize_lists: each of n_clusters ivf::lists rounds its capacity up to the next multiple of
  // 1024 (large lists, align_max=1024 non-conservative). Worst case: each cluster wastes 1023 rows.
  size_t resize_lists_rows = n_rows + 1023 * n_clusters;
  size_t resize_lists_dev  = resize_lists_rows * bytes_per_row;

  // Peak device = max(placeholder_dev, resize_lists_dev); they don't overlap (placeholder freed
  // before resize_lists).
  // resize_lists_dev is also the permanent net delta (it stays as index list data after extend).
  size_t total_dev = std::max(placeholder_dev, resize_lists_dev);

  std::cout << "ivf_pq::build_knn::build::extend::placeholder_list " << to_mib(placeholder_dev) << " MiB"
            << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::vec_batches_buf " << to_mib(batch_size * dim * dtype_size)
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::new_data_labels " << to_mib(n_rows * sizeof(uint32_t))
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::cluster_centers " << to_mib(n_clusters * dim * sizeof(float))
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::new_vectors_residual "
            << to_mib(batch_size * rot_dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::flat_compute_residuals_tmp "
            << to_mib(batch_size * dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::extend::pq_codes " << to_mib(resize_lists_dev) << " MiB"
            << std::endl;
  std::cout << "memuse_ivf_pq_extend: "
            << " GiB, workspace: " << to_gib(total_ws)
            << " GiB, large_workspace: " << to_gib(new_data_labels_large)
            << " GiB, device: " << to_gib(total_dev) << " GiB" << std::endl;

  return {.host            = 0,
          .pinned          = 0,
          .workspace       = total_ws,
          .large_workspace = new_data_labels_large,
          .managed         = 0,
          .device          = total_dev};
}

// Estimate peak memory allocated per source during ivf_pq_build.cuh::build().
MemUsage memuse_ivf_pq_build(size_t n_rows,
                             size_t n_rows_train,
                             size_t dim,
                             size_t pq_dim,
                             size_t pq_len,
                             size_t pq_book_size,
                             size_t n_clusters,
                             size_t index_size,
                             size_t dtype_size,
                             bool codebook_per_cluster,
                             const raft::resources& handle)
{
  size_t rot_dim = pq_dim * pq_len;

  // trainset buffers on big_memory_resource (workspace or large_workspace).
  size_t trainset_ws = n_rows_train * dim * sizeof(float);
  size_t trainset_tmp_ws =
    (dtype_size != sizeof(float)) ? n_rows_train * dim * dtype_size : static_cast<size_t>(0);
  size_t trainset_base = trainset_ws + trainset_tmp_ws;

  // sample_rows: raft::matrix::sample_rows gathers from host dataset into trainset.
  // gather uses 2 pinned ping-pong buffers (32768 KiB each) and a host index array.
  // device peak: (6*n_excess + n_rows_train) * int64 — upper-bound with n_excess = n_rows_train.
  constexpr size_t kGatherBufBytes = 32768ULL * 1024;
  size_t gather_batch       = std::min(((kGatherBufBytes / dim + 31) / 32) * 32, n_rows_train);
  size_t sample_rows_pinned = 2 * gather_batch * dim * dtype_size;
  size_t sample_rows_host   = n_rows_train * sizeof(int64_t);
  size_t sample_rows_dev    = 7 * n_rows_train * sizeof(int64_t);

  // cluster_centers_buf: allocated on workspace, persists from its range through train_pq.
  size_t cluster_centers_ws = n_clusters * dim * sizeof(float);

  // k_means_clustering range: workspace delta during kmeans::fit (build_hierarchical).
  // cluster_centers_buf is already live; the delta is the fit-internal temps.
  // n_mesoclusters = min(n_clusters, sqrt(n_clusters)+0.5) as in the implementation.
  size_t n_mesoclusters =
    std::min(n_clusters, static_cast<size_t>(std::sqrt(static_cast<double>(n_clusters)) + 0.5));
  // large_workspace: mc_trainset_buf = mesocluster_size_max * dim * float (concurrent with
  // trainset)
  size_t mc_size_max  = (2 * n_rows_train + n_mesoclusters - 1) / n_mesoclusters;
  size_t kmeans_large = mc_size_max * dim * sizeof(float);
  // device: mesocluster_labels_buf on managed memory
  size_t kmeans_fit_dev = n_rows_train * sizeof(uint32_t);

  // labels range: n_rows_train * uint32_t on big_memory_resource (workspace or large_workspace).
  size_t labels_large = n_rows_train * sizeof(uint32_t);

  // Upper bound: always-fused path (fusedL2NN, all rows in one batch).
  // Fit delta: dataset_norm_buf (4) + inner labels (4) + minClusterAndDist (8) + mutex (4)
  size_t kmeans_fit_ws =
    n_rows_train * (sizeof(float) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(int));
  // Predict delta: cur_dataset_norm (4) + minClusterAndDist (8) + mutex (4)
  size_t kmeans_predict_ws = n_rows_train * (sizeof(float) + sizeof(uint64_t) + sizeof(int));

  // train_pq range: workspace delta during PQ codebook training.
  // cluster_centers_buf + labels still live; new buffers are for the codebook training.
  size_t pq_centers_size = (codebook_per_cluster ? n_clusters : pq_dim) * pq_book_size * pq_len;
  size_t train_pq_ws     = pq_centers_size * sizeof(float);  // pq_centers_tmp
  if (!codebook_per_cluster) {
    size_t pq_n_rows = std::min(size_t(256) * pq_book_size, n_rows_train);
    train_pq_ws += pq_n_rows * pq_len * sizeof(float);  // sub_trainset
    train_pq_ws += pq_n_rows * sizeof(uint32_t);        // sub_labels
  } else {
    size_t max_cluster_size = n_rows_train / std::max(n_clusters, size_t(1));
    train_pq_ws += n_rows_train * index_size;                     // indices_buf
    train_pq_ws += max_cluster_size * pq_dim * sizeof(uint32_t);  // pq_labels
    train_pq_ws += max_cluster_size * rot_dim * sizeof(float);    // rot_vectors
  }
  // managed (PER_CLUSTER): cluster_sizes + offsets_buf in train_per_cluster
  size_t train_pq_managed =
    codebook_per_cluster ? (n_clusters + (n_clusters + 1)) * sizeof(uint32_t) : 0;

  // extend range
  size_t pq_bits_val = 0;
  for (size_t b = pq_book_size; b > 1; b >>= 1)
    pq_bits_val++;
  auto ext = memuse_ivf_pq_extend(
    n_rows, dim, rot_dim, n_clusters, index_size, dtype_size, pq_dim, pq_bits_val);

  // --- Peak computation (worst-case upper bounds) ---
  // workspace upper bound: Case A (trainset + labels on workspace) — always larger than Case B.
  size_t workspace_internal =
    trainset_base + cluster_centers_ws +
    std::max({kmeans_fit_ws, labels_large + kmeans_predict_ws, labels_large + train_pq_ws});
  // large_workspace upper bound: Case B (trainset on large_workspace) — always larger than Case A.
  size_t train_large = trainset_base + std::max(kmeans_large, labels_large);
  size_t total_ws    = std::max(workspace_internal, ext.workspace);
  size_t total_large = std::max(train_large, ext.large_workspace);

  size_t total_host    = std::max(sample_rows_host, ext.host);
  size_t total_pinned  = sample_rows_pinned;
  size_t total_managed = std::max(train_pq_managed, ext.managed);
  size_t total_dev     = std::max({sample_rows_dev, kmeans_fit_dev, ext.device});

  // --- Print per-range estimates in NVTX range order ---
  std::cout << "ivf_pq::build_knn::build::trainset workspace_or_large_workspace "
            << to_mib(trainset_ws) << " MiB" << std::endl;
  if (dtype_size != sizeof(float)) {
    std::cout << "ivf_pq::build_knn::build::trainset_tmp workspace_or_large_workspace "
              << to_mib(trainset_tmp_ws) << " MiB" << std::endl;
    std::cout << "ivf_pq::build_knn::build::sample_rows_other_types host " << to_mib(sample_rows_host)
              << " MiB, pinned " << to_mib(sample_rows_pinned) << " MiB, device "
              << to_mib(sample_rows_dev) << " MiB" << std::endl;
  } else {
    std::cout << "ivf_pq::build_knn::build::sample_rows_float host " << to_mib(sample_rows_host)
              << " MiB, pinned " << to_mib(sample_rows_pinned) << " MiB, device "
              << to_mib(sample_rows_dev) << " MiB" << std::endl;
  }
  std::cout << "ivf_pq::build_knn::build::cluster_centers_buf workspace " << to_mib(cluster_centers_ws)
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::kmeans_clustering workspace " << to_mib(kmeans_fit_ws)
            << " MiB, large_workspace " << to_mib(kmeans_large) << " MiB, device "
            << to_mib(kmeans_fit_dev) << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::labels workspace_or_large_workspace "
            << to_mib(labels_large) << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::build::kmeans_predict workspace " << to_mib(kmeans_predict_ws) << " MiB"
            << std::endl;
  std::cout << "memuse_ivf_pq_build: "
            << "host: " << to_gib(total_host) << " GiB"
            << ", pinned: " << to_gib(total_pinned) << " GiB"
            << ", workspace: " << to_gib(total_ws) << " GiB"
            << ", large_workspace: " << to_gib(total_large) << " GiB"
            << ", managed: " << to_gib(total_managed) << " GiB"
            << ", device: " << to_gib(total_dev) << " GiB" << std::endl;

  return {.host            = total_host,
          .pinned          = total_pinned,
          .workspace       = total_ws,
          .large_workspace = total_large,
          .managed         = total_managed,
          .device          = total_dev};
}

// Estimate peak workspace for one outer batch of ivf_pq::search()
static MemUsage memuse_ivf_pq_search(size_t n_outer,  // coarse batch (outer loop)
                                     size_t n_inner,  // fine worker batch (inner loop)
                                     size_t dim_ext,
                                     size_t rot_dim,
                                     size_t n_lists,
                                     size_t n_probes,
                                     size_t top_k)
{
  constexpr size_t kF  = sizeof(float);
  constexpr size_t kH  = 2;  // sizeof(half) — CAGRA uses CUDA_R_16F for coarse search
  constexpr size_t kU4 = sizeof(uint32_t);

  // ivf_pq::build_knn::search::gemm_queries  (half precision: coarse_search_dtype=CUDA_R_16F)
  size_t gemm_queries = n_outer * dim_ext * kH;
  // ivf_pq::build_knn::search::rot_queries  (always float)
  size_t rot_queries = n_outer * rot_dim * kF;

  // ivf_pq::build_knn::search::select_clusters::qc_distances  (half precision)
  size_t qc_distances = n_outer * n_lists * kH;

  // Worker allocations use n_inner (get_max_fine_batch_size may reduce batch below n_outer).
  // ivf_pq::build_knn::search::worker::distances_buf  (fused: topk_len = n_probes * top_k)
  size_t distances_buf = n_inner * n_probes * top_k * kF;
  // ivf_pq::build_knn::search::worker::neighbors_buf  (fused path; same topk_len, uint32_t)
  size_t neighbors_buf = n_inner * n_probes * top_k * kU4;
  size_t worker   = distances_buf + neighbors_buf;

  size_t total_ws = gemm_queries + rot_queries + std::max(qc_distances, worker);

  std::cout << "ivf_pq::build_knn::search::gemm_queries " << to_mib(gemm_queries) << " MiB"
            << std::endl;
  std::cout << "ivf_pq::build_knn::search::rot_queries " << to_mib(rot_queries) << " MiB"
            << std::endl;
  std::cout << "ivf_pq::build_knn::search::select_clusters::qc_distances " << to_mib(qc_distances)
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::search::worker::distances_buf " << to_mib(distances_buf)
            << " MiB" << std::endl;
  std::cout << "ivf_pq::build_knn::search::worker::neighbors_buf " << to_mib(neighbors_buf)
            << " MiB" << std::endl;
  std::cout << "memuse_ivf_pq_search: workspace: " << to_gib(total_ws) << " GiB"  << std::endl;

  return {.workspace = total_ws};
}

// Estimate peak memory allocated per source during cagra_build.cuh::build_knn_graph().
MemUsage memuse_build_knn_graph_by_ivf_pq(size_t n_rows,
                                          size_t n_rows_train,
                                          size_t dim,
                                          size_t node_degree,
                                          size_t gpu_top_k,
                                          size_t dtype_size,
                                          size_t pq_dim,
                                          size_t pq_len,
                                          size_t pq_bits,
                                          size_t pq_book_size,
                                          size_t n_lists,
                                          size_t n_probes,
                                          size_t max_queries,
                                          bool codebook_per_cluster,
                                          const raft::resources& handle)
{
  size_t top_k   = node_degree + 1;
  size_t rot_dim = pq_dim * pq_len;

  // Phase 1: IVF-PQ build temporaries
  auto build = memuse_ivf_pq_build(n_rows,
                                   n_rows_train,
                                   dim,
                                   pq_dim,
                                   pq_len,
                                   pq_book_size,
                                   n_lists,
                                   sizeof(int64_t),  // IVF-PQ only instantiates int64_t index type
                                   dtype_size,
                                   codebook_per_cluster,
                                   handle);

  // IVF-PQ compressed index lives in device memory across both phases:
  //   compressed vectors: n_rows * ceil(pq_dim * pq_bits / 8)
  size_t compressed_vectors = n_rows * ((pq_dim * pq_bits + 7) / 8);

  // Phase 2: upper-bound batch sizes — workspace assumed unconstrained, so both batches
  // saturate at max_queries (n_inner ≤ n_outer ≤ max_queries).
  size_t dim_ext = ((dim + 8) / 8) * 8;  // L2 extends dim by 1, rounded to multiple of 8
  // account for the max possible values of n_outer and n_inner
  size_t n_outer = max_queries;
  size_t n_inner = max_queries;

  // Phase 2: outer I/O buffers in workspace_mr (from build_knn_graph in cagra_build.cuh)
  size_t outer_search_ws = n_outer * gpu_top_k * sizeof(float)      // distances
                           + n_outer * gpu_top_k * sizeof(int64_t)  // neighbors
                           + n_outer * top_k * sizeof(float)        // refined_distances
                           + n_outer * top_k * sizeof(int64_t)      // refined_neighbors
                           + n_outer * dim * dtype_size;            // vec_batches

  size_t outer_search_host = n_outer * gpu_top_k * sizeof(int64_t)  // neighbors_host
                             + n_outer * dim * dtype_size           // queries_host
                             + n_outer * top_k * sizeof(int64_t)    // refined_neighbors_host
                             + n_outer * top_k * sizeof(float);     // refined_distances_host

  // Phase 2: internal ivf_pq::search workspace (on top of outer buffers, same workspace_mr)
  auto ivf_pq_search =
    memuse_ivf_pq_search(n_outer, n_inner, dim_ext, rot_dim, n_lists, n_probes, gpu_top_k);
  size_t search_ws = outer_search_ws + ivf_pq_search.workspace;

  size_t total_host    = std::max(build.host, outer_search_host);
  size_t total_pinned  = build.pinned;
  size_t total_ws      = std::max(build.workspace, search_ws);
  size_t total_large   = build.large_workspace;
  size_t total_managed = build.managed;
  // compressed_vectors persists through both phases; build.device (flat_compute_residuals tmp) is
  // concurrent with the index during extend(), then freed before the search phase starts.
  size_t total_dev = compressed_vectors + build.device;

  std::cout << "ivf_pq::build_knn::outer_search " << to_mib(outer_search_ws) << " MiB workspace, "
            << to_mib(outer_search_host) << " MiB host" << std::endl;
  std::cout << "memuse_build_knn_graph_by_ivf_pq: host: " << to_gib(total_host)
            << " GiB, pinned: " << to_gib(total_pinned)
            << " GiB, workspace: " << to_gib(total_ws)
            << " GiB, large_workspace: " << to_gib(total_large)
            << " GiB, managed: " << to_gib(total_managed) << " GiB, device: " << to_gib(total_dev)
            << " GiB" << std::endl;

  return {.host            = total_host,
          .pinned          = total_pinned,
          .workspace       = total_ws,
          .large_workspace = total_large,
          .managed         = total_managed,
          .device          = total_dev};
}

// Estimate peak memory allocated per source during cagra_build.cuh::build() (IVF-PQ path).
//
// Memory layout by phase:
//   Phase 1 – build_knn_graph (IVF-PQ build + KNN search):
//     host:          knn_graph (N*D_in*idx) + memuse_build_knn_graph_by_ivf_pq.host
//     workspace:     memuse_build_knn_graph_by_ivf_pq.workspace
//     large_workspace: memuse_build_knn_graph_by_ivf_pq.large_workspace
//     managed:       memuse_build_knn_graph_by_ivf_pq.managed
//     device:        IVF-PQ compressed index (freed when build_knn_graph returns)
//   Phase 2 – optimize:
//     host:          knn_graph (N*D_in*idx) + cagra_graph (N*D_out*idx) + memuse_optimize.host
//     workspace:     memuse_optimize.workspace
//     large_workspace: memuse_optimize.large_workspace (d_input_graph = knn graph on device)
//     device:        memuse_optimize.device
//   Phase 3 – attach_dataset_on_build (index_params default: true):
//     device:        dataset copy (N*dim*T) — sequential with phases 1&2, so peak = max of all
//     three
//   Note: IVF-PQ index on device is freed before optimize → device peak = max(phase1, phase2,
//   phase3).
//         knn_graph freed after optimize (knn_graph.reset() in build()).
MemUsage memuse_cagra_build(size_t n_rows,
                            size_t n_rows_train,
                            size_t dim,
                            size_t graph_degree,
                            size_t intermediate_degree,
                            size_t index_size,
                            size_t dtype_size,
                            size_t gpu_top_k,
                            size_t pq_dim,
                            size_t pq_len,
                            size_t pq_bits,
                            size_t pq_book_size,
                            size_t n_lists,
                            size_t n_probes,
                            size_t max_queries,
                            bool codebook_per_cluster,
                            bool guarantee_connectivity,
                            bool attach_dataset,
                            const raft::resources& handle)
{
  // Persistent host buffers: knn_graph alive across both phases; cagra_graph added before optimize
  size_t knn_graph_host   = n_rows * intermediate_degree * index_size;
  size_t cagra_graph_host = n_rows * graph_degree * index_size;

  // Phase 1: build_knn_graph (IVF-PQ build + KNN search)
  auto bkg = memuse_build_knn_graph_by_ivf_pq(n_rows,
                                              n_rows_train,
                                              dim,
                                              intermediate_degree,
                                              gpu_top_k,
                                              dtype_size,
                                              pq_dim,
                                              pq_len,
                                              pq_bits,
                                              pq_book_size,
                                              n_lists,
                                              n_probes,
                                              max_queries,
                                              codebook_per_cluster,
                                              handle);

  // Phase 2: optimize (knn_graph and cagra_graph both alive; IVF-PQ index already freed)
  auto opt =
    memuse_optimize(n_rows, graph_degree, intermediate_degree, index_size, guarantee_connectivity);

  // Phase 3: attach_dataset_on_build — the index constructor always copies the CAGRA graph
  // host→device (n_rows * graph_degree * index_size). The dataset copy depends on whether the
  // caller's buffer is device-accessible (HMM/pinned → non-owning, 0 bytes; regular host →
  // n_rows * dim * dtype_size). Estimate only the graph copy as the guaranteed allocation.
  size_t graph_dev = attach_dataset ? n_rows * graph_degree * index_size : 0;

  // Peak per source (all three phases are sequential, so take max across phases)
  size_t total_host =
    std::max(knn_graph_host + bkg.host, knn_graph_host + cagra_graph_host + opt.host);
  size_t total_ws      = std::max(bkg.workspace, opt.workspace);
  size_t total_large   = std::max(bkg.large_workspace, opt.large_workspace);
  size_t total_managed = std::max(bkg.managed, opt.managed);
  size_t total_dev     = std::max({bkg.device, opt.device, graph_dev});
  // RAFT pre-allocates a ~256 MiB pinned staging pool at raft::resources creation time
  constexpr size_t kRaftPinnedPoolBytes = 256ULL * 1024 * 1024;
  size_t total_pinned                   = kRaftPinnedPoolBytes;

  std::cout << "cagra_build::knn_graph host " << to_gib(knn_graph_host) << " GiB" << std::endl;
  std::cout << "cagra_build::cagra_graph host " << to_gib(cagra_graph_host) << " GiB" << std::endl;
  if (attach_dataset) {
    std::cout << "cagra_build::attach_dataset::graph device " << to_gib(graph_dev) << " GiB"
              << std::endl;
  }
  std::cout << "memuse_cagra_build: host: " << to_gib(total_host) << " GiB, "
            << "pinned: " << to_gib(total_pinned) << " GiB, "
            << "workspace: " << to_gib(total_ws) << " GiB, "
            << "large_workspace: " << to_gib(total_large) << " GiB, "
            << "managed: " << to_gib(total_managed) << " GiB, "
            << "device: " << to_gib(total_dev) << " GiB" << std::endl;

  return {.host            = total_host,
          .pinned          = total_pinned,
          .workspace       = total_ws,
          .large_workspace = total_large,
          .managed         = total_managed,
          .device          = total_dev};
}

// High-level overload: extracts IVF-PQ parameters from cagra::index_params directly.
// Only the IVF-PQ build path is supported; returns a zeroed MemUsage for other paths.
MemUsage memuse_cagra_build(size_t n_rows,
                            size_t dim,
                            size_t dtype_size,
                            const cuvs::neighbors::cagra::index_params& cagra_params,
                            const raft::resources& handle)
{
  if (!std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
        cagra_params.graph_build_params)) {
    RAFT_LOG_WARN("memuse_cagra_build: IVF-PQ build params not set, skipping estimate");
    return {};
  }

  const auto& pq = std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
    cagra_params.graph_build_params);

  const size_t pq_dim       = pq.build_params.pq_dim;
  const size_t pq_bits      = pq.build_params.pq_bits;
  const size_t pq_len       = (dim + pq_dim - 1) / pq_dim;
  const size_t pq_book_size = static_cast<size_t>(1) << pq_bits;
  const size_t n_lists      = pq.build_params.n_lists;
  const bool codebook_per_cluster =
    pq.build_params.codebook_kind == cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
  // Mirror ivf_pq_build.cuh: trainset_ratio = n_rows / max(fraction*n_rows, n_lists),
  // then n_rows_train = n_rows / trainset_ratio.  Two integer divisions, NOT a direct max().
  const size_t trainset_max = std::max<size_t>(
    static_cast<size_t>(pq.build_params.kmeans_trainset_fraction * n_rows), n_lists);
  const size_t trainset_ratio = std::max<size_t>(1, n_rows / trainset_max);
  const size_t n_rows_train   = n_rows / trainset_ratio;
  const size_t node_degree    = cagra_params.intermediate_graph_degree;
  const size_t top_k          = node_degree + 1;
  const size_t gpu_top_k      = std::min<size_t>(
    std::max<size_t>(static_cast<size_t>(node_degree * pq.refinement_rate), top_k), n_rows);
  const size_t n_probes    = pq.search_params.n_probes;
  const size_t max_queries = pq.search_params.max_internal_batch_size;

  return memuse_cagra_build(n_rows,
                            n_rows_train,
                            dim,
                            cagra_params.graph_degree,
                            cagra_params.intermediate_graph_degree,
                            sizeof(uint32_t),
                            dtype_size,
                            gpu_top_k,
                            pq_dim,
                            pq_len,
                            pq_bits,
                            pq_book_size,
                            n_lists,
                            n_probes,
                            max_queries,
                            codebook_per_cluster,
                            cagra_params.guarantee_connectivity,
                            cagra_params.attach_dataset_on_build,
                            handle);
}

}  // namespace cuvs::neighbors::cagra::helpers
