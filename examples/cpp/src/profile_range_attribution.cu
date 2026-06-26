/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Example showing off RESPONSIBLE-RANGE attribution in
 * raft::memory_tracking_resources.
 *
 * Two behaviours are demonstrated that a naive, sampling-based monitor cannot
 * attribute correctly:
 *
 *   1. NESTED ranges.  An allocation made inside an inner range is charged to
 *      that inner range AND to every enclosing (outer) range.  In the CSV the
 *      `alloc_range` column carries the full "outer#id > inner#id" path, so
 *      plot_mem.py shows both the inner and the outer ranges.
 *
 *   2. OVERLAPPING ranges.  A buffer is ALLOCATED while range "produce" is
 *      active but FREED later, after range "produce" has already ended and
 *      range "consume" has been pushed.  Because the responsible range is
 *      captured at ALLOCATION time and remembered per address, the free is
 *      charged back to "produce" -- not to whatever range happens to be active
 *      when the deallocation runs.  This is the common shape for asynchronous /
 *      cross-phase allocations (e.g. a workspace produced in one phase and
 *      released in the next).
 *
 * Output: writes "range_attribution_stats.csv" in the working directory.
 *
 * NOTE: the `nvtx_range` / `alloc_range` columns are only populated when this
 * example is compiled with NVTX enabled (configure with -DRAFT_NVTX=ON).  The
 * allocation counters are recorded regardless of that flag.
 */

#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/memory_tracking_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <optional>
#include <thread>

namespace nvtx = raft::common::nvtx;

namespace {

constexpr std::size_t MiB = std::size_t{1024} * 1024;

}  // namespace

void nesting_range_example(raft::resources& res)
{
  auto stream = raft::resource::get_cuda_stream(res);
  // The tracked device resource (memory_tracking_resources installed it as the
  // current device resource), so these device_buffers are recorded as "device".
  auto dev_mr = rmm::mr::get_current_device_resource_ref();

  nvtx::range outer{"outer 256 + 64 + 128MB"};

  // 256 MiB allocated directly under build_graph.
  rmm::device_buffer graph(50 * MiB, stream, dev_mr);

  {
    nvtx::range inner{"inner 64MB"};
    rmm::device_buffer tmp(100 * MiB, stream, dev_mr);
  }  // tmp freed here, still inside build_graph

  {
    nvtx::range inner{"inner 128MB"};
    rmm::device_buffer tmp(25 * MiB, stream, dev_mr);
  }  // tmp freed here, still inside build_graph
}

int main()
{
  raft::resources res;

  // Wrap `res` so every reachable memory resource is tracked; CSV rows are
  // written from a background thread as events happen.
  raft::memory_tracking_resources tracked(res, "range_attribution_stats.csv");

  nesting_range_example(tracked);

  std::cout << "Wrote allocation statistics to range_attribution_stats.csv\n"
            << "Visualize with: python plot_mem_per_range.py range_attribution_stats.csv\n";
  return 0;
}
