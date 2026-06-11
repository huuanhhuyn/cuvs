/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal example that exercises raft::memory_tracking_resources.
 *
 * It performs a series of deliberately-sized allocations against every memory
 * source the tracker wraps (host, pinned, managed, device, workspace and
 * large_workspace) and wraps each in an NVTX range so the produced CSV can be
 * grouped by range (see examples/cpp/plot_mem.py).
 *
 * Output: writes "prof_stats.csv" in the working directory.
 *
 * NOTE: the `nvtx_range` column is only populated when this example is compiled
 * with NVTX enabled (configure cuvs / this example with -DRAFT_NVTX=ON).  The
 * allocation counters are recorded regardless of that flag.
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/memory_tracking_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>

namespace nvtx = raft::common::nvtx;

namespace {

constexpr std::size_t MiB = std::size_t{1024} * 1024;

// Number of float elements that occupy `mib` mebibytes.
constexpr std::size_t floats_for(std::size_t mib) { return (mib * MiB) / sizeof(float); }

// Give the 1ms sampler a moment to record a row while each allocation is live.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }

}  // namespace

void example(raft::resources& res)
{
  nvtx::range function_scope{"example"};

  // --- Host: a small matrix and a larger vector ---------------------------
  {
    nvtx::range r{"1. expect 40 KB host matrix"};
    auto matrix = raft::make_host_matrix<float>(res, 10, 1024); 
    settle();
  }
  {
    nvtx::range r{"2. expect 10 GB host vector"};
    auto vector = raft::make_host_vector<float>(res, floats_for(10*1024));
    settle();
  }

  // --- Pinned host memory --------------------------------------------------
  {
    nvtx::range r{"3. expect 128 MiB pinned vector"};
    auto pinned = raft::make_pinned_vector<float>(res, floats_for(128));
    settle();
  }

  // --- Managed (unified) memory -------------------------------------------
  {
    nvtx::range r{"4. expect 128 MiB managed vector"};
    auto managed = raft::make_managed_vector<float>(res, floats_for(64));
    settle();
  }

  // --- Device memory -------------------------------------------------------
  {
    nvtx::range r{"5. expect 128 MiB device vector"};
    auto device = raft::make_device_vector<float>(res, floats_for(192));
    settle();
  }

  // --- Workspace and large-workspace resources -----------------------------
  // These are stream-ordered device allocators; allocate raw device_buffers
  // directly from the (tracked) workspace resource refs.
  auto stream = raft::resource::get_cuda_stream(res);
  {
    nvtx::range r{"6. expect 80 MiB workspace buffer"};
    rmm::device_buffer ws_buf(80 * MiB, stream, raft::resource::get_workspace_resource_ref(res));
    //stream.synchronize();
    settle();
  }
  {
    nvtx::range r{"7. expect 320 MiB large_workspace buffer"};
    rmm::device_buffer lws_buf(
      320 * MiB, stream, raft::resource::get_large_workspace_resource_ref(res));
    //stream.synchronize();
    settle();
  }
}

int main()
{
  raft::resources res;

  // Wrap `res` so every reachable memory resource is tracked, sampling at 1ms
  // and logging CSV rows from a background thread.
  raft::memory_tracking_resources tracked(res, "prof_stats.csv", std::chrono::milliseconds(1));

  example(tracked);

  std::cout << "Wrote allocation statistics to prof_stats.csv\n. Visualize with: python plot_mem.py prof_stats.csv\n";
  return 0;
}
