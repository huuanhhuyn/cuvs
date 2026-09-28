/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Run:
 *   ./CUVS_KMEANS_BENCH [--benchmark_filter=<regex>]
 *
 * TODO:
 *   1: Bench on multi GPU
 *   2: Run on GEMM-1NN PR
 *   3: Example and Nsys analysis
 */

#include <benchmark/benchmark.h>

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/random/make_blobs.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_fp16.h>
#include <omp.h>

#include <cstdint>
#include <type_traits>

namespace cuvs::bench {

template <typename DataT, typename IdxT>
static raft::device_matrix<DataT, IdxT> make_blobs(const raft::resources& res,
                                                    IdxT n_rows,
                                                    IdxT n_cols,
                                                    int n_clusters)
{
  if constexpr (std::is_same_v<DataT, half>) {
    // raft::random::make_blobs does not support half (box_muller_transform uses
    // constexpr Type(2.0) which requires constexpr __half ctor — not available).
    // Generate float blobs and cast element-wise.
    auto X_f   = raft::make_device_matrix<float, IdxT>(res, n_rows, n_cols);
    auto labels = raft::make_device_vector<IdxT, IdxT>(res, n_rows);
    raft::random::make_blobs<float, IdxT>(res, X_f.view(), labels.view(), (IdxT)n_clusters);
    auto X_h = raft::make_device_matrix<half, IdxT>(res, n_rows, n_cols);
    raft::linalg::map(res,
                      X_h.view(),
                      [] __device__(float v) { return __float2half(v); },
                      raft::make_const_mdspan(X_f.view()));
    raft::resource::sync_stream(res);
    return X_h;
  } else {
    auto X      = raft::make_device_matrix<DataT, IdxT>(res, n_rows, n_cols);
    auto labels = raft::make_device_vector<IdxT, IdxT>(res, n_rows);
    raft::random::make_blobs<DataT, IdxT>(res, X.view(), labels.view(), (IdxT)n_clusters);
    raft::resource::sync_stream(res);
    return X;
  }
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------

struct KmeansBenchParams {
  int                             n_rows;
  int                             n_cols;
  int                             n_clusters;
  cuvs::distance::DistanceType    metric;
};

static const KmeansBenchParams kParams[] = {
  // L2Expanded
  {20000,  128,  50,    cuvs::distance::DistanceType::L2Expanded},
  {20000,  128,  1000,  cuvs::distance::DistanceType::L2Expanded},
  {100000, 128,  50000, cuvs::distance::DistanceType::L2Expanded},
  {20000, 768,  4000, cuvs::distance::DistanceType::L2Expanded},
  {20000, 3072, 4000, cuvs::distance::DistanceType::L2Expanded},
  // InnerProduct and CosineExpanded are not supported?
  //{20000,  768,  4000,  cuvs::distance::DistanceType::InnerProduct},
  //{20000,  768,  4000,  cuvs::distance::DistanceType::CosineExpanded},
};

static void AddParams(benchmark::internal::Benchmark* b)
{
  for (const auto& p : kParams) {
    b->Args({p.n_rows, p.n_cols, p.n_clusters, static_cast<int>(p.metric)});
  }
  b->ArgNames({"n_rows", "n_cols", "n_clusters", "metric"});
  b->Unit(benchmark::kMillisecond);
  b->UseRealTime();
  b->Iterations(3);
}

// ---------------------------------------------------------------------------
// Non-balanced kmeans
// ---------------------------------------------------------------------------

template <typename DataT, typename IdxT>
static void BM_KmeansFit(benchmark::State& state)
{
  const IdxT n_rows     = static_cast<IdxT>(state.range(0));
  const IdxT n_cols     = static_cast<IdxT>(state.range(1));
  const int  n_clusters = static_cast<int>(state.range(2));
  const auto metric     = static_cast<cuvs::distance::DistanceType>(state.range(3));

  raft::resources res;
  auto X         = make_blobs<DataT, IdxT>(res, n_rows, n_cols, n_clusters);
  auto centroids = raft::make_device_matrix<float, IdxT>(res, n_clusters, n_cols);

  cuvs::cluster::kmeans::params p;
  p.n_clusters          = n_clusters;
  p.max_iter            = 10;
  p.n_init              = 1;
  p.init                = cuvs::cluster::kmeans::params::KMeansPlusPlus;
  p.tol                 = 1e-4f;
  p.oversampling_factor = 2.0;
  p.rng_state.seed      = 42;
  p.metric              = metric;

  float inertia = 0;
  IdxT  n_iter  = 0;
  auto  X_view  = raft::make_const_mdspan(X.view());

  for (auto _ : state) {
    cuvs::cluster::kmeans::fit(res,
                               p,
                               X_view,
                               std::optional<raft::device_vector_view<const float, IdxT>>{},
                               centroids.view(),
                               raft::make_host_scalar_view<float>(&inertia),
                               raft::make_host_scalar_view<IdxT>(&n_iter));
    raft::resource::sync_stream(res);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_rows));
  state.counters["n_rows"]     = static_cast<double>(n_rows);
  state.counters["n_cols"]     = static_cast<double>(n_cols);
  state.counters["n_clusters"] = n_clusters;
}

template <typename DataT, typename IdxT>
static void BM_KmeansPredict(benchmark::State& state)
{
  const IdxT n_rows     = static_cast<IdxT>(state.range(0));
  const IdxT n_cols     = static_cast<IdxT>(state.range(1));
  const int  n_clusters = static_cast<int>(state.range(2));
  const auto metric     = static_cast<cuvs::distance::DistanceType>(state.range(3));

  raft::resources res;
  auto X         = make_blobs<DataT, IdxT>(res, n_rows, n_cols, n_clusters);
  auto centroids = raft::make_device_matrix<float, IdxT>(res, n_clusters, n_cols);
  auto labels    = raft::make_device_vector<IdxT, IdxT>(res, n_rows);

  cuvs::cluster::kmeans::params p;
  p.n_clusters          = n_clusters;
  p.max_iter            = 1;
  p.n_init              = 1;
  p.init                = cuvs::cluster::kmeans::params::KMeansPlusPlus;
  p.tol                 = 1e-4f;
  p.oversampling_factor = 2.0;
  p.rng_state.seed      = 42;
  p.metric              = metric;

  float inertia = 0;
  IdxT  n_iter  = 0;
  auto  X_view  = raft::make_const_mdspan(X.view());

  // Train once outside the loop.
  cuvs::cluster::kmeans::fit(res,
                              p,
                              X_view,
                              std::optional<raft::device_vector_view<const float, IdxT>>{},
                              centroids.view(),
                              raft::make_host_scalar_view<float>(&inertia),
                              raft::make_host_scalar_view<IdxT>(&n_iter));
  raft::resource::sync_stream(res);

  float pred_inertia = 0;
  auto  C_view       = raft::make_const_mdspan(centroids.view());

  for (auto _ : state) {
    cuvs::cluster::kmeans::predict(res,
                                   p,
                                   X_view,
                                   std::optional<raft::device_vector_view<const float, IdxT>>{},
                                   C_view,
                                   labels.view(),
                                   /*normalize_weight=*/false,
                                   raft::make_host_scalar_view<float>(&pred_inertia));
    raft::resource::sync_stream(res);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_rows));
  state.counters["n_rows"]     = static_cast<double>(n_rows);
  state.counters["n_cols"]     = static_cast<double>(n_cols);
  state.counters["n_clusters"] = n_clusters;
}

// ---------------------------------------------------------------------------
// Balanced kmeans
// ---------------------------------------------------------------------------
template <typename DataT>
static void BM_KmeansBalancedFit(benchmark::State& state)
{
  const int64_t n_rows     = state.range(0);
  const int64_t n_cols     = state.range(1);
  const int     n_clusters = static_cast<int>(state.range(2));
  const auto    metric     = static_cast<cuvs::distance::DistanceType>(state.range(3));

  raft::resources res;
  auto X         = make_blobs<DataT, int>(res, n_rows, n_cols, n_clusters);
  auto centroids = raft::make_device_matrix<float, int64_t>(res, n_clusters, n_cols);

  cuvs::cluster::kmeans::balanced_params p;
  p.n_iters = 10;
  p.metric  = metric;

  auto X_view = raft::make_device_matrix_view<const DataT, int64_t>(
    X.data_handle(), n_rows, n_cols);

  for (auto _ : state) {
    cuvs::cluster::kmeans::fit(res, p, X_view, centroids.view());
    raft::resource::sync_stream(res);
  }

  state.SetItemsProcessed(state.iterations() * n_rows);
  state.counters["n_rows"]     = static_cast<double>(n_rows);
  state.counters["n_cols"]     = static_cast<double>(n_cols);
  state.counters["n_clusters"] = n_clusters;
  state.counters["dtype"]      = sizeof(DataT);
}

template <typename DataT>
static void BM_KmeansBalancedPredict(benchmark::State& state)
{
  const int64_t n_rows     = state.range(0);
  const int64_t n_cols     = state.range(1);
  const int     n_clusters = static_cast<int>(state.range(2));
  const auto    metric     = static_cast<cuvs::distance::DistanceType>(state.range(3));

  raft::resources res;
  auto X         = make_blobs<DataT, int>(res, n_rows, n_cols, n_clusters);
  auto centroids = raft::make_device_matrix<float, int64_t>(res, n_clusters, n_cols);
  auto labels    = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);

  cuvs::cluster::kmeans::balanced_params p;
  p.n_iters = 1;
  p.metric  = metric;

  auto X_view = raft::make_device_matrix_view<const DataT, int64_t>(
    X.data_handle(), n_rows, n_cols);

  cuvs::cluster::kmeans::fit(res, p, X_view, centroids.view());
  raft::resource::sync_stream(res);

  auto C_view = raft::make_const_mdspan(centroids.view());

  for (auto _ : state) {
    cuvs::cluster::kmeans::predict(res, p, X_view, C_view, labels.view());
    raft::resource::sync_stream(res);
  }

  state.SetItemsProcessed(state.iterations() * n_rows);
  state.counters["n_rows"]     = static_cast<double>(n_rows);
  state.counters["n_cols"]     = static_cast<double>(n_cols);
  state.counters["n_clusters"] = n_clusters;
  state.counters["dtype"]      = sizeof(DataT);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Standard kmeans — int32 index
BENCHMARK(BM_KmeansFit<float, int>)->Apply(AddParams)->Name("BM_KmeansFit/f32/i32");
BENCHMARK(BM_KmeansPredict<float, int>)->Apply(AddParams)->Name("BM_KmeansPredict/f32/i32");

// Standard kmeans — int64 index
BENCHMARK(BM_KmeansFit<float, int64_t>)->Apply(AddParams)->Name("BM_KmeansFit/f32/i64");
BENCHMARK(BM_KmeansPredict<float, int64_t>)->Apply(AddParams)->Name("BM_KmeansPredict/f32/i64");

// Balanced kmeans — float32 input
BENCHMARK(BM_KmeansBalancedFit<float>)->Apply(AddParams)->Name("BM_KmeansBalancedFit/f32");
BENCHMARK(BM_KmeansBalancedPredict<float>)->Apply(AddParams)->Name("BM_KmeansBalancedPredict/f32");

// Balanced kmeans — float16 (half) input, float32 centroids
BENCHMARK(BM_KmeansBalancedFit<half>)->Apply(AddParams)->Name("BM_KmeansBalancedFit/f16");
BENCHMARK(BM_KmeansBalancedPredict<half>)->Apply(AddParams)->Name("BM_KmeansBalancedPredict/f16");

}  // namespace cuvs::bench

BENCHMARK_MAIN();
