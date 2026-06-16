/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
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
  em_optimize(n_rows, graph_degree, intermediate_degree, index_size, mst_optimize);

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

// Estimate peak memory allocated per source during ivf_pq_build.cuh::extend().
//
// Memory layout by phase:
//   Upfront device (default allocator):
//     placeholder_list: round_up(N + 31*K, 1024) * bytes_per_row  [freed before resize_lists]
//       where bytes_per_row = round_up(pq_dim*pq_bits, 128)/8 + sizeof(int64_t)
//   Phase 1 – label prediction:
//     workspace: new_data_labels (N*4) + cluster_centers (K*dim*4) + vec_batches_buf (batch*dim*T)
//   Phase 2 – fill codes (cluster_centers freed; new_data_labels + vec_batches_buf still alive):
//     workspace: new_data_labels (N*4) + orig_list_sizes (K*4) + vec_batches_buf (batch*dim*T)
//              + new_vectors_residual (batch*rot_dim*4) + flat_compute_residuals_tmp (batch*dim*4)
//   After fill codes:
//     resize_lists: (N + 1023*K) * bytes_per_row  [worst-case upper bound; permanent index data]
//       Each ivf::list rounds capacity to next power-of-2 (small lists) or next multiple of 1024
//       (large lists, align_max=1024 non-conservative). Worst case: each cluster wastes 1023 rows.
//   Note: vec_batches_buf is 1 buffer without a stream pool (no prefetch), 2 with.
//         idx_batches is null (new_indices=nullptr on initial build), so no idx buffer.
//         If workspace is too small, new_data_labels spills to large_workspace.
MemUsage em_ivf_pq_extend(size_t n_rows,
                          size_t dim,
                          size_t rot_dim,
                          size_t n_clusters,
                          size_t index_size,
                          size_t dtype_size,
                          size_t pq_dim,
                          size_t pq_bits)
{
  size_t batch_size = std::min(n_rows, size_t(65536));

  // Host: two std::vectors for cluster size bookkeeping (new_cluster_sizes + old_cluster_sizes)
  size_t host = 2 * n_clusters * sizeof(uint32_t);

  // new_data_labels is allocated via raft large-workspace resource (not the batch workspace).
  size_t large_workspace = n_rows * sizeof(uint32_t);  // new_data_labels

  size_t ws_label = n_clusters * dim * sizeof(float)    // cluster_centers
                    + batch_size * dim * dtype_size;    // vec_batches_buf (1 buffer, no prefetch)

  // flat_compute_residuals_tmp uses batches_mr (workspace), not the default device allocator.
  size_t ws_fill = n_clusters * sizeof(uint32_t)             // orig_list_sizes
                   + batch_size * dim * dtype_size           // vec_batches_buf
                   + batch_size * rot_dim * sizeof(float)    // new_vectors_residual
                   + batch_size * dim * sizeof(float);       // flat_compute_residuals_tmp

  size_t workspace = std::max(ws_label, ws_fill);

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
  // 1024 (for large lists) or next power-of-2 (for small lists, min 32, max 1024).
  // Worst-case upper bound: each cluster wastes up to 1023 extra rows (align_max - 1).
  size_t resize_lists_dev = (n_rows + 1023 * n_clusters) * bytes_per_row;

  // Peak device = max(placeholder_dev, resize_lists_dev); they don't overlap (placeholder freed
  // before resize_lists). With the worst-case resize_lists formula, resize_lists_dev dominates.
  // resize_lists_dev is also the permanent net delta (it stays as index list data after extend).
  size_t device = std::max(placeholder_dev, resize_lists_dev);  // being conservative for the worst case

  std::cout << "ivf_pq::build::extend::placeholder_list " << to_mib(placeholder_dev) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::vec_batches_buf " << to_mib(batch_size * dim * dtype_size) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::new_data_labels " << to_mib(n_rows * sizeof(uint32_t)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::cluster_centers " << to_mib(n_clusters * dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::orig_list_sizes " << to_mib(n_clusters * sizeof(uint32_t)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::new_vectors_residual " << to_mib(batch_size * rot_dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::flat_compute_residuals_tmp " << to_mib(batch_size * dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::extend::resize_lists " << to_mib(resize_lists_dev) << " MiB" << std::endl;
  std::cout << "em_ivf_pq_extend: host: " << to_gib(host) << " GiB, workspace: " << to_gib(workspace) << " GiB, large_workspace: " << to_gib(large_workspace) << " GiB, device: " << to_gib(device) << " GiB" << std::endl;

  return {.host            = host,
          .pinned          = 0,
          .workspace       = workspace,
          .large_workspace = large_workspace,
          .managed         = 0,
          .device          = device};
}

// Estimate peak memory allocated per source during ivf_pq_build.cuh::build().
//
// Memory layout:
//   Training phase (sequential with extend — take max per source):
//     large_workspace: trainset (n_train*dim*4) + trainset_tmp (n_train*dim*T, non-float)
//                    + labels (n_train*4)
//     workspace:  cluster_centers_buf (K*dim*4) + PQ codebook training buffers:
//       PER_SUBSPACE: pq_centers_tmp + sub_trainset + sub_labels
//       PER_CLUSTER:  pq_centers_tmp + indices_buf + pq_labels + rot_vectors
//     managed (PER_CLUSTER only): cluster_sizes (K*4) + offsets_buf ((K+1)*4)
//   Extend phase: em_ivf_pq_extend (called after training, trainset freed first)
MemUsage em_ivf_pq_build(size_t n_rows,
                         size_t n_rows_train,
                         size_t dim,
                         size_t pq_dim,
                         size_t pq_len,
                         size_t pq_book_size,
                         size_t n_clusters,
                         size_t index_size,
                         size_t dtype_size,
                         bool codebook_per_cluster)
{
  size_t rot_dim = pq_dim * pq_len;

  // Training: trainset + trainset_tmp persist throughout the entire training phase.
  // labels is allocated AFTER kmeans::fit returns (so labels and mc_trainset_buf are exclusive).
  // mc_trainset_buf (kmeans_large_ws, computed below) is allocated DURING kmeans::fit while
  // trainset is still alive — they are concurrent, not alternative.
  // Peak large_workspace = trainset_base + max(mc_trainset_buf, labels)   (computed after kmeans_large_ws)
  size_t trainset_base_large = n_rows_train * dim * sizeof(float)  // trainset (float)
                               + (dtype_size != sizeof(float) ? n_rows_train * dim * dtype_size
                                                              : static_cast<size_t>(0));  // trainset_tmp
  size_t labels_large = n_rows_train * sizeof(uint32_t);

  // Training: workspace holds cluster_centers_buf + PQ codebook training buffers
  size_t pq_centers_size = (codebook_per_cluster ? n_clusters : pq_dim) * pq_book_size * pq_len;
  size_t train_ws        = n_clusters * dim * sizeof(float)  // cluster_centers_buf
                    +
                    pq_centers_size * sizeof(float);  // pq_centers_tmp (per_subset or per_cluster)

  if (!codebook_per_cluster) {
    // train_per_subset: sub_trainset + sub_labels (per subspace, not concurrent)
    size_t pq_n_rows = std::min(size_t(256) * pq_book_size, n_rows_train);
    train_ws += pq_n_rows * pq_len * sizeof(float);  // sub_trainset
    train_ws += pq_n_rows * sizeof(uint32_t);        // sub_labels
  } else {
    // train_per_cluster: indices_buf + pq_labels + rot_vectors (max_cluster_size = avg)
    size_t max_cluster_size = n_rows_train / std::max(n_clusters, size_t(1));
    train_ws += n_rows_train * index_size;                     // indices_buf
    train_ws += max_cluster_size * pq_dim * sizeof(uint32_t);  // pq_labels
    train_ws += max_cluster_size * rot_dim * sizeof(float);    // rot_vectors
  }

  // Managed memory (PER_CLUSTER only): cluster_sizes + offsets_buf (small)
  size_t managed = codebook_per_cluster ? (n_clusters + (n_clusters + 1)) * sizeof(uint32_t) : 0;

  // Extend phase — sequential with training, take max per source
  // pq_bits = log2(pq_book_size); pq_book_size is a power of 2
  size_t pq_bits_val = 0;
  for (size_t b = pq_book_size; b > 1; b >>= 1) pq_bits_val++;
  auto ext = em_ivf_pq_extend(n_rows, dim, rot_dim, n_clusters, index_size, dtype_size, pq_dim, pq_bits_val);

  // sample_rows (float or other-types) gathers from the original host-resident dataset.
  // raft::matrix::detail::gather uses 2 pinned ping-pong buffers for H2D overlap:
  //   buffer_size = 32768 * 1024 bytes (from gather.cuh), max_batch_size = round_up(buf/dim, 32)
  constexpr size_t kGatherBufBytes = 32768ULL * 1024;
  size_t gather_batch              = std::min(((kGatherBufBytes / dim + 31) / 32) * 32, n_rows_train);
  size_t sample_rows_pinned        = 2 * gather_batch * dim * dtype_size;  // out_tmp1 + out_tmp2
  // indices_host copies the sampled indices to host so gather_buff() can access them.
  size_t sample_rows_host = n_rows_train * sizeof(int64_t);

  // excess_subsample peak device: rnd_idx + linear_idx + keys_out + values_out ≈ 4 device vectors.
  // n_excess = ceil(log(1 - n_train/N) / log(1 - 1/N)) + max(0.1*n_train, 100), capped at N.
  double frac     = std::min(static_cast<double>(n_rows_train) / static_cast<double>(n_rows), 0.9999);
  auto n_excess   = static_cast<size_t>(
    std::ceil(std::log(1.0 - frac) / std::log(1.0 - 1.0 / static_cast<double>(n_rows))));
  n_excess       += std::max(static_cast<size_t>(static_cast<double>(n_rows_train) * 0.1), static_cast<size_t>(100));
  n_excess         = std::min(n_excess, n_rows);
  // Peak: rnd_idx(n_excess*8) + linear_idx(n_excess*8) + CUB_MergeSort_ws(n_excess*16)
  //     + keys_out(n_excess*8) + values_out_old(n_excess*8) + values_out_new(n_rows_train*8)
  //   The move-assignment `values_out = make_device_vector(res, n_samples)` briefly keeps both
  //   the old n_excess-sized vector and the new n_rows_train-sized vector alive at the same time.
  size_t sample_rows_device = (6 * n_excess + n_rows_train) * sizeof(int64_t);

  // k_means_clustering: balanced kmeans via build_hierarchical.
  // n_mesoclusters = min(n_clusters, sqrt(n_clusters)+0.5) as in the implementation.
  size_t n_mesoclusters = std::min(n_clusters,
    static_cast<size_t>(std::sqrt(static_cast<double>(n_clusters)) + 0.5));
  // large_workspace: mc_trainset_buf = mesocluster_size_max_balanced * dim * sizeof(float)
  //   where mesocluster_size_max_balanced = ceil(2*n_rows_train / n_mesoclusters).
  size_t mc_size_max      = (2 * n_rows_train + n_mesoclusters - 1) / n_mesoclusters;
  size_t kmeans_large_ws  = mc_size_max * dim * sizeof(float);
  // workspace: cluster_centers_buf (persists) + dataset_norm_buf + labels_inner
  //          + minClusterAndDistance (KeyValuePair<int32_t,float>=8B) + CUB_fused_mutex (4B/row)
  size_t kmeans_ws = n_clusters * dim * sizeof(float)    // cluster_centers_buf (always live)
                   + n_rows_train * sizeof(float)         // dataset_norm_buf
                   + n_rows_train * sizeof(uint32_t)      // labels (inner build_hierarchical)
                   + n_rows_train * sizeof(uint64_t)      // minClusterAndDistance (int32+float = 8B)
                   + n_rows_train * sizeof(int);          // CUB fusedL2NN mutex workspace
  // device: mesocluster_labels_buf on rmm::mr::managed_memory_resource (shows as "device")
  size_t kmeans_device = n_rows_train * sizeof(uint32_t);

  size_t host            = std::max(sample_rows_host, ext.host);
  size_t pinned          = sample_rows_pinned;
  size_t workspace       = std::max({train_ws, kmeans_ws, ext.workspace});
  // trainset_base persists; mc_trainset_buf and labels are mutually exclusive on top of it.
  size_t train_large_ws  = trainset_base_large + std::max(kmeans_large_ws, labels_large);
  size_t large_workspace = std::max(train_large_ws, ext.large_workspace);
  size_t device          = std::max({sample_rows_device, kmeans_device, ext.device});

  std::cout << "ivf_pq::build::trainset " << to_mib(n_rows_train * dim * sizeof(float)) << " MiB" << std::endl;
  if (dtype_size != sizeof(float)) {
    std::cout << "ivf_pq::build::trainset_tmp " << to_mib(n_rows_train * dim * dtype_size) << " MiB" << std::endl;
  }
  if (dtype_size == sizeof(float)) {
    std::cout << "ivf_pq::build::sample_rows_float host " << to_mib(sample_rows_host) << " MiB, pinned " << to_mib(sample_rows_pinned) << " MiB, device " << to_mib(sample_rows_device) << " MiB" << std::endl;
  } else {
    std::cout << "ivf_pq::build::sample_rows_other_types host " << to_mib(sample_rows_host) << " MiB, pinned " << to_mib(sample_rows_pinned) << " MiB, device " << to_mib(sample_rows_device) << " MiB" << std::endl;
  }
  std::cout << "ivf_pq::build::k_means_clustering workspace " << to_mib(kmeans_ws) << " MiB, large_workspace " << to_mib(kmeans_large_ws) << " MiB, device " << to_mib(kmeans_device) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::cluster_centers_buf " << to_mib(n_clusters * dim * sizeof(float)) << " MiB" << std::endl;
  std::cout << "ivf_pq::build::labels " << to_mib(n_rows_train * sizeof(uint32_t)) << " MiB" << std::endl;
  std::cout << "em_ivf_pq_build: host: " << to_gib(host) << " GiB, pinned: " << to_gib(pinned) << " GiB, workspace: " << to_gib(workspace) << " GiB, large_workspace: " << to_gib(large_workspace) << " GiB, managed: " << to_gib(managed) << " GiB, device: " << to_gib(device) << " GiB" << std::endl;

  return {.host            = host,
          .pinned          = pinned,
          .workspace       = workspace,
          .large_workspace = large_workspace,
          .managed         = managed,
          .device          = device};
}

// Estimate peak memory allocated per source during cagra_build.cuh::build_knn_graph().
//
// Memory layout:
//   Phase 1 – ivf_pq::build (em_ivf_pq_build):
//     large_workspace: trainset + labels
//     workspace:       build temporaries
//     device:          IVF-PQ compressed index (persists through search phase)
//   Phase 2 – ivf_pq::search loop (sequential with build):
//     workspace_mr:    distances (B*gpu_top_k*4) + neighbors (B*gpu_top_k*8)
//                    + refined_distances (B*top_k*4) + refined_neighbors (B*top_k*8)
//                    + vec_batches (B*dim*T)
//     host:            neighbors_host (B*gpu_top_k*8) + queries_host (B*dim*T)
//                    + refined_neighbors_host (B*top_k*8) + refined_distances_host (B*top_k*4)
//   workspace_mr spills to large_workspace when free workspace < kMinWorkspaceRatio * desired.
MemUsage em_build_knn_graph_by_ivf_pq(size_t n_rows,
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
                                      size_t max_queries,
                                      bool codebook_per_cluster)
{
  size_t top_k   = node_degree + 1;
  size_t rot_dim = pq_dim * pq_len;

  // Phase 1: IVF-PQ build temporaries
  auto build = em_ivf_pq_build(n_rows,
                               n_rows_train,
                               dim,
                               pq_dim,
                               pq_len,
                               pq_book_size,
                               n_lists,
                               sizeof(int64_t),  // IVF-PQ only instantiates int64_t index type
                               dtype_size,
                               codebook_per_cluster);

  // IVF-PQ compressed index lives in device memory across both phases:
  //   compressed vectors: n_rows * ceil(pq_dim * pq_bits / 8)
  //   cluster centers:    n_lists * round_up(dim+1, 8) * sizeof(float)
  //   rotation matrix:    rot_dim * dim * sizeof(float)
  size_t index_device = n_rows * ((pq_dim * pq_bits + 7) / 8) +
                        n_lists * ((dim + 8) & ~size_t(7)) * sizeof(float) +
                        rot_dim * dim * sizeof(float);

  // Phase 2: search I/O buffers (workspace_mr = workspace unless free_space is too small)
  size_t search_ws = max_queries * gpu_top_k * sizeof(float)      // distances
                     + max_queries * gpu_top_k * sizeof(int64_t)  // neighbors
                     + max_queries * top_k * sizeof(float)        // refined_distances
                     + max_queries * top_k * sizeof(int64_t)      // refined_neighbors
                     + max_queries * dim * dtype_size;            // vec_batches

  // Phase 2: host staging buffers (live across all search batches)
  size_t search_host = max_queries * gpu_top_k * sizeof(int64_t)  // neighbors_host
                       + max_queries * dim * dtype_size           // queries_host
                       + max_queries * top_k * sizeof(int64_t)    // refined_neighbors_host
                       + max_queries * top_k * sizeof(float);     // refined_distances_host

  size_t host            = std::max(build.host, search_host);
  size_t workspace       = std::max(build.workspace, search_ws);
  size_t large_workspace = build.large_workspace;
  size_t managed         = build.managed;
  // index_device persists through both phases; build.device (flat_compute_residuals tmp) is
  // concurrent with the index during extend(), then freed before the search phase starts.
  size_t device = index_device + build.device;

  std::cout << "cagra_build::ivf_pq::buffers " << to_mib(search_ws - max_queries * dim * dtype_size) << " MiB workspace, " << to_mib(search_host) << " MiB host" << std::endl;
  std::cout << "cagra_build::ivf_pq::vec_batches " << to_mib(max_queries * dim * dtype_size) << " MiB" << std::endl;
  std::cout << "em_build_knn_graph_by_ivf_pq: host: " << to_gib(host) << " GiB, workspace: " << to_gib(workspace) << " GiB, large_workspace: " << to_gib(large_workspace) << " GiB, managed: " << to_gib(managed) << " GiB, device: " << to_gib(device) << " GiB" << std::endl;

  return {.host            = host,
          .pinned          = 0,
          .workspace       = workspace,
          .large_workspace = large_workspace,
          .managed         = managed,
          .device          = device};
}

// Estimate peak memory allocated per source during graph_core::optimize().
//
// Memory layout by phase:
//   Phase 1 – mst_optimization (if guarantee_connectivity):
//     host:  outer mst_graph (N*D_out*idx) + mst_graph_num_edges (N*4)
//          + inner mst_graph (N*D_out*idx) + 7 work vectors (7*N*idx)   <- peak host
//     device: d_mst_graph (N*D_out*idx) + 8 vectors (8*N*idx)          <- peak device
//   Phase 2 – prune_graph_gpu:
//     large_workspace: d_input_graph (N*D_in*idx, full knn)             <- peak large_workspace
//     workspace:       d_output_graph (2*batch*D_out*idx)
//   Phase 3 – make_reverse + merge_graph_gpu:
//     large_workspace: d_rev_graph (N*D_out*idx)  [sequential with prune, so not peak]
//     workspace:       d_rev_graph_count (N*4)
//                    + d_output_graph (2*batch*D_out*idx)
//                    + d_mst_graph (2*batch*D_out*idx, if mst)
//                    + d_mst_graph_num_edges (2*batch*4, if mst)        <- peak workspace
//     device:          d_dest_nodes (N*idx)  [host-accessible graph path, non-mst peak]
MemUsage em_optimize(size_t n_rows,
                     size_t graph_degree,
                     size_t intermediate_degree,
                     size_t index_size,
                     bool guarantee_connectivity)
{
  size_t batch_size = std::min(static_cast<size_t>(256 * 1024), n_rows);

  // Peak host: outer + inner mst arrays all live simultaneously (if mst)
  size_t host = 0;
  if (guarantee_connectivity) {
    host += n_rows * graph_degree * index_size;  // mst_graph (outer, in optimize())
    std::cout << "optimize::mst_graph " << to_mib(n_rows * graph_degree * index_size) << " MiB" << std::endl;
    host += n_rows * sizeof(uint32_t);           // mst_graph_num_edges (outer)
    std::cout << "optimize::mst_graph_num_edges " << to_mib(n_rows * sizeof(uint32_t)) << " MiB" << std::endl;
    host += n_rows * graph_degree * index_size;  // mst_graph (inner, in mst_optimization())
    std::cout << "optimize::mst_optimization::mst_graph " << to_mib(n_rows * graph_degree * index_size) << " MiB" << std::endl;
    host += 7 * n_rows *
            index_size;  // outgoing/incoming_max/num_edges, label, cluster_size, candidate_edges
    std::cout << "optimize::mst_optimization::work_vectors " << to_mib(7 * n_rows * index_size) << " MiB" << std::endl;
    std::cout << "optimize::mst_optimization::d_mst_graph " << to_mib((graph_degree + 1) * n_rows * index_size) << " MiB" << std::endl;
    std::cout << "optimize::mst_optimization::d_work_vectors " << to_mib(7 * n_rows * index_size) << " MiB" << std::endl;
  }

  // Peak large_workspace: prune loads full knn graph; d_rev_graph allocated after prune returns
  size_t large_workspace = n_rows * intermediate_degree * index_size;  // d_input_graph

  // Peak workspace: merge stage (d_rev_graph_count persists + batch iterators)
  size_t workspace = n_rows * sizeof(uint32_t);             // d_rev_graph_count
  workspace += 2 * batch_size * graph_degree * index_size;  // d_output_graph (2 batches)
  if (guarantee_connectivity) {
    workspace += 2 * batch_size * graph_degree * index_size;  // d_mst_graph (2 batches)
    workspace += 2 * batch_size * sizeof(uint32_t);           // d_mst_graph_num_edges (2 batches)
  }

  // Peak device (default allocator):
  //   mst case: d_mst_graph + 8 vectors in mst_optimization (freed before prune)
  //   no-mst case: d_dest_nodes in make_reverse_graph_gpu (host-accessible output path)
  size_t device = 0;
  if (guarantee_connectivity) {
    device += n_rows * graph_degree * index_size;  // d_mst_graph
    device += 8 * n_rows * index_size;             // d_mst_graph_num_edges + 7 work vectors
  } else {
    device += n_rows * index_size;  // d_dest_nodes
    std::cout << "optimize::make_reverse_graph_gpu::d_dest_nodes " << to_mib(n_rows * index_size) << " MiB" << std::endl;
    std::cout << "optimize::make_reverse_graph_gpu::dest_nodes " << to_mib(n_rows * index_size) << " MiB" << std::endl;
  }

  std::cout << "optimize::prune::d_input_graph " << to_mib(n_rows * intermediate_degree * index_size) << " MiB" << std::endl;
  std::cout << "optimize::prune::d_output_graph " << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
  std::cout << "optimize::d_rev_graph " << to_mib(n_rows * graph_degree * index_size) << " MiB" << std::endl;
  std::cout << "optimize::d_rev_graph_count " << to_mib(n_rows * sizeof(uint32_t)) << " MiB" << std::endl;
  std::cout << "optimize::merge::d_output_graph " << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
  if (guarantee_connectivity) {
    std::cout << "optimize::merge::d_mst_graph " << to_mib(2 * batch_size * graph_degree * index_size) << " MiB" << std::endl;
    std::cout << "optimize::merge::d_mst_graph_num_edges " << to_mib(2 * batch_size * sizeof(uint32_t)) << " MiB" << std::endl;
  }
  std::cout << "em_optimize: host: " << to_gib(host) << " GiB, "
            << "pinned: " << to_gib(0) << " GiB, "
            << "workspace: " << to_gib(workspace) << " GiB, "
            << "large_workspace: " << to_gib(large_workspace) << " GiB, "
            << "managed: " << to_gib(0) << " GiB, "
            << "device: " << to_gib(device) << " GiB" << std::endl;

  return {.host            = host,
          .pinned          = 0,
          .workspace       = workspace,
          .large_workspace = large_workspace,
          .managed         = 0,
          .device          = device};
}

// Estimate peak memory allocated per source during cagra_build.cuh::build() (IVF-PQ path).
//
// Memory layout by phase:
//   Phase 1 – build_knn_graph (IVF-PQ build + KNN search):
//     host:          knn_graph (N*D_in*idx) + em_build_knn_graph_by_ivf_pq.host
//     workspace:     em_build_knn_graph_by_ivf_pq.workspace
//     large_workspace: em_build_knn_graph_by_ivf_pq.large_workspace
//     managed:       em_build_knn_graph_by_ivf_pq.managed
//     device:        IVF-PQ compressed index (freed when build_knn_graph returns)
//   Phase 2 – optimize:
//     host:          knn_graph (N*D_in*idx) + cagra_graph (N*D_out*idx) + em_optimize.host
//     workspace:     em_optimize.workspace
//     large_workspace: em_optimize.large_workspace (d_input_graph = knn graph on device)
//     device:        em_optimize.device
//   Phase 3 – attach_dataset_on_build (index_params default: true):
//     device:        dataset copy (N*dim*T) — sequential with phases 1&2, so peak = max of all
//     three
//   Note: IVF-PQ index on device is freed before optimize → device peak = max(phase1, phase2,
//   phase3).
//         knn_graph freed after optimize (knn_graph.reset() in build()).
MemUsage em_cagra_build(size_t n_rows,
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
                        size_t max_queries,
                        bool codebook_per_cluster,
                        bool guarantee_connectivity,
                        bool attach_dataset)
{
  // Persistent host buffers: knn_graph alive across both phases; cagra_graph added before optimize
  size_t knn_graph_host   = n_rows * intermediate_degree * index_size;
  size_t cagra_graph_host = n_rows * graph_degree * index_size;

  // Phase 1: build_knn_graph (IVF-PQ build + KNN search)
  auto bkg = em_build_knn_graph_by_ivf_pq(n_rows,
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
                                          max_queries,
                                          codebook_per_cluster);

  // Phase 2: optimize (knn_graph and cagra_graph both alive; IVF-PQ index already freed)
  auto opt =
    em_optimize(n_rows, graph_degree, intermediate_degree, index_size, guarantee_connectivity);

  // Phase 3: attach_dataset_on_build — copies full dataset to device (sequential with phases 1&2)
  size_t dataset_device = attach_dataset ? n_rows * dim * dtype_size : static_cast<size_t>(0);

  // Peak per source (all three phases are sequential, so take max across phases)
  size_t host = std::max(knn_graph_host + bkg.host, knn_graph_host + cagra_graph_host + opt.host);
  size_t workspace       = std::max(bkg.workspace, opt.workspace);
  size_t large_workspace = std::max(bkg.large_workspace, opt.large_workspace);
  size_t managed         = std::max(bkg.managed, opt.managed);
  size_t device          = std::max({bkg.device, opt.device, dataset_device});
  // RAFT pre-allocates a ~256 MiB pinned staging pool at raft::resources creation time
  constexpr size_t kRaftPinnedPoolBytes = 256ULL * 1024 * 1024;
  size_t pinned                         = kRaftPinnedPoolBytes;

  std::cout << "cagra_build::knn_graph " << to_mib(knn_graph_host) << " MiB" << std::endl;
  std::cout << "cagra_build::cagra_graph " << to_mib(cagra_graph_host) << " MiB" << std::endl;
  if (attach_dataset) {
    std::cout << "!!! cagra_build::attach_dataset_on_build " << to_mib(dataset_device) << " MiB" << std::endl;
  }
  std::cout << "em_cagra_build: host: " << to_gib(host) << " GiB, "
            << "pinned: " << to_gib(pinned) << " GiB, "
            << "workspace: " << to_gib(workspace) << " GiB, "
            << "large_workspace: " << to_gib(large_workspace) << " GiB, "
            << "managed: " << to_gib(managed) << " GiB, "
            << "device: " << to_gib(device) << " GiB" << std::endl;

  return {.host            = host,
          .pinned          = pinned,
          .workspace       = workspace,
          .large_workspace = large_workspace,
          .managed         = managed,
          .device          = device};
}

// High-level overload: extracts IVF-PQ parameters from cagra::index_params directly.
// Only the IVF-PQ build path is supported; returns a zeroed MemUsage for other paths.
MemUsage em_cagra_build(size_t n_rows,
                        size_t dim,
                        size_t dtype_size,
                        const cuvs::neighbors::cagra::index_params& cagra_params)
{
  if (!std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
        cagra_params.graph_build_params)) {
    std::cout << "em_cagra_build: IVF-PQ build params not set, skipping estimate" << std::endl;
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
  const size_t trainset_max   = std::max<size_t>(
    static_cast<size_t>(pq.build_params.kmeans_trainset_fraction * n_rows), n_lists);
  const size_t trainset_ratio = std::max<size_t>(1, n_rows / trainset_max);
  const size_t n_rows_train   = n_rows / trainset_ratio;
  const size_t node_degree = cagra_params.intermediate_graph_degree;
  const size_t top_k       = node_degree + 1;
  const size_t gpu_top_k   = std::min<size_t>(
    std::max<size_t>(static_cast<size_t>(node_degree * pq.refinement_rate), top_k), n_rows);
  const size_t max_queries = pq.search_params.max_internal_batch_size;

  return em_cagra_build(n_rows,
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
                        max_queries,
                        codebook_per_cluster,
                        cagra_params.guarantee_connectivity,
                        cagra_params.attach_dataset_on_build);
}

}  // namespace cuvs::neighbors::cagra::helpers
