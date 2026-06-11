# memory-estimation code map

| # | Function / symbol | Estimator location | Allocation block it estimates | What it estimates |
|---|---|---|---|---|
| 1 | `optimize_workspace_size` | [cagra_helpers.cpp:15](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L15) | [`optimize()` graph_core.cuh:1704](../../cpp/src/neighbors/detail/cagra/graph_core.cuh#L1704) — `mst_graph` [:1745](../../cpp/src/neighbors/detail/cagra/graph_core.cuh#L1745), `d_rev_graph` [:1774](../../cpp/src/neighbors/detail/cagra/graph_core.cuh#L1774), `d_input_graph` [:1003](../../cpp/src/neighbors/detail/cagra/graph_core.cuh#L1003) | Working memory of CAGRA graph optimization (MST + prune + reverse + combine stages) |
| 2 | `ivf_pq_build_mem_usage` | [cagra_helpers.cpp:76](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L76) | IVF‑PQ search buffers [cagra_build.cuh:1734](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1734) + kmeans trainset [ivf_pq_build.cuh:1283](../../cpp/src/neighbors/ivf_pq/ivf_pq_build.cuh#L1283) | CAGRA-via-IVF‑PQ build footprint (compressed dataset + graph + workspace + kmeans trainset) |
| 3 | `cagra_build_mem_usage` | [cagra_helpers.cpp:112](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L112) | [`build()` cagra_build.cuh:2198](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L2198) — `knn_graph` [:2258](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L2258), `cagra_graph` [:2309](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L2309) | Top-level CAGRA build estimate; dispatches by graph-build algo |
| 4 | `ivf_pq::helpers::compressed_dataset_size` | [ivf_pq_index.cu:822](../../cpp/src/neighbors/ivf_pq_index.cu#L822) · decl [ivf_pq.hpp:3305](../../cpp/include/cuvs/neighbors/ivf_pq.hpp#L3305) | [`owning_impl` ctor ivf_pq_index.cu:211](../../cpp/src/neighbors/ivf_pq_index.cu#L211) (`pq_centers_`/`centers_`/`centers_rot_`/`rotation_matrix_`) + per-list data on `extend` | On-device size of a populated IVF‑PQ index |
| 5 | `nn_descent::has_enough_device_memory` | [nn_descent.cu:22](../../cpp/src/neighbors/nn_descent.cu#L22) · decl [nn_descent.hpp:534](../../cpp/include/cuvs/neighbors/nn_descent.hpp#L534) | [`GNND` ctor buffers nn_descent.cuh:1349](../../cpp/src/neighbors/detail/nn_descent.cuh#L1349) (probe itself allocs at [nn_descent.cu:28](../../cpp/src/neighbors/nn_descent.cu#L28)) | Feasibility probe (tries to allocate NN‑descent buffers) |
| 6 | `struct ace_memory_requirements` | [cagra_build.cuh:810](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L810) | `build_ace`: `partition_labels` [:1230](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1230), `search_graph` [:1315](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1315), `sub_dataset` [:1343](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1343), `sub_search_graph` [:1397](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1397) | Container for ACE host/GPU requirement breakdown |
| 7 | ACE tunable constants | [cagra_build.cuh:822](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L822) | [`build_ace` cagra_build.cuh:1100](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1100) (whole build) | Safety/overhead factors for ACE model |
| 8 | `ace_check_use_disk_mode` | [cagra_build.cuh:829](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L829) | per-partition `sub_dataset` [:1343](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1343) + final `search_graph` [:1315](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1315) | ACE host+GPU model; decides disk vs in‑memory |
| 9 | `ace_validate_disk_mode_partitions` | [cagra_build.cuh:943](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L943) | [`build_ace` cagra_build.cuh:1100](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L1100) (per-partition allocs) | Re-estimates disk-mode footprint; grows partition count to fit |
| 10 | `get_available_memory` | [hnsw.hpp:1175](../../cpp/src/neighbors/detail/hnsw.hpp#L1175) | — *(measures free memory; not an allocation)* | Available host/device memory (honors user GiB caps) |
| 11 | `hnsw::build` monostate heuristic | [hnsw.hpp:1493](../../cpp/src/neighbors/detail/hnsw.hpp#L1493) | [`cagra::build` cagra_build.cuh:2198](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L2198) (the build it guards) | Compares required vs available → in‑memory vs ACE |
| 12 | `hnsw::from_cagra` spill estimate | [hnsw.hpp:1255](../../cpp/src/neighbors/detail/hnsw.hpp#L1255) | [`HierarchicalNSW` alloc hnsw.hpp:985](../../cpp/src/neighbors/detail/hnsw.hpp#L985) (in-mem `data_level0_memory_`) | HNSW host footprint → spill index to disk? |
| — | `cuvs::util::get_free_host_memory` | [host_memory.hpp](../../cpp/include/cuvs/util/host_memory.hpp) | — *(primitive)* | Primitive: host free bytes (device side uses `rmm::available_device_memory()`) |

## Formulas / breakdowns

### 1. [`optimize_workspace_size`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L15)`(n_rows, graph_degree, intermediate_degree, index_size, mst_optimize)`
`batch_size = min(256K, n_rows)`

| Component | Memory | Side |
|---|---|---|
| MST (optional) | `n_rows·index_size` + `2·n_rows·graph_degree·index_size` + `7·n_rows·index_size` + `(gd-1)²·index_size` | host |
| Prune | `n_rows·intermediate_degree·index_size` (`d_input_graph`) + detour counts + `2·batch·graph_degree·index_size` | device |
| Reverse graph | `n_rows·graph_degree·index_size` + `n_rows·4` + `n_rows·index_size` | device |
| Combine | `n_rows·4` (+ histogram); device `2·batch·gd·index_size` (×2 more if MST) | host + device |
| **Return** | `total_host = mst+combine`, `total_dev = max(prune, rev+combine)`, plus the two `*_fixed` | — |

### 2. [`ivf_pq_build_mem_usage`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L76)`(...)`
| Term | Formula |
|---|---|
| `dataset_gpu_mem` | [`compressed_dataset_size(...)`](../../cpp/src/neighbors/ivf_pq_index.cu#L822) (#4) |
| `graph_host_mem` | `n_rows·(graph_degree + intermediate_graph_degree)·4` |
| workspace | [`optimize_workspace_size(...)`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L15) (#1) |
| `kmeans_gpu_mem` | `kmeans_n_rows · dim · 4` (trainset float buffer, freed before extend) |
| **total_host** | `graph_host_mem + host_workspace + 2e9` (IVF‑PQ search slack) |
| **total_dev** | `max(kmeans_gpu_mem, dataset_gpu_mem, gpu_workspace) + 1e9` |

### 3. [`cagra_build_mem_usage`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L112)`(...)` — dispatch
| Branch | Estimate | Notes |
|---|---|---|
| IVF‑PQ params | delegates to [`ivf_pq_build_mem_usage`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L76) | precise-ish |
| NN‑descent params | `dataset·dtype + graph·4 + 2e9`; `total_dev = total_host` | marked `TODO proper estimate` |
| iterative / else | same rough formula as NN‑descent | marked `TODO proper estimate` |

### 4. [`compressed_dataset_size`](../../cpp/src/neighbors/ivf_pq_index.cu#L822)`(...)` — populated IVF‑PQ index (device)
`kIndexGroupSize=32`, `kIndexGroupVecLen=16`, `pq_chunk = (16·8)/pq_bits`

| Array | Formula |
|---|---|
| `pq_centers` | `pq_len · 2^pq_bits · pq_dim · 4` |
| `pq_dataset` | `ceildiv(n_rows,32)·32 · ceildiv(pq_dim, pq_chunk) · 16` |
| `indices` | `n_rows · 8` |
| `rotation_matrix` | `rot_dim² · 4` |
| `list_offsets` | `(n_lists+1) · 8` |
| `list_sizes` | `n_lists · 8` |
| `centers` | `n_lists · dim_ext · 4` |
| `centers_rot` | `n_lists · rot_dim · 4` |
| **Return** | sum of all above |

> Note: this is the *predicted* analogue of the IVF‑PQ `owning_impl` constructor's eager allocations ([ivf_pq_index.cu:211](../../cpp/src/neighbors/ivf_pq_index.cu#L211): `pq_centers_`, `centers_`, `centers_rot_`, `rotation_matrix_`) — e.g. ~38 MiB of metadata at `dim=1536`, `n_lists≈2.4k` — separate from the per-list encoded data which grows during `extend`.

### 5. [`has_enough_device_memory`](../../cpp/src/neighbors/nn_descent.cu#L22)`(...)` — probe, not formula
Attempts to allocate the NN‑descent working set and returns `true`, or `false` on `bad_alloc`/`logic_error`:
- `d_data` `[n_rows × dim]` half
- `l2_norms` `[n_rows]` float
- `graph_buffer` `[n_rows · idx_size · DEGREE_ON_DEVICE]` uint32
- `dists_buffer` `[n_rows × DEGREE_ON_DEVICE]` float
- `d_locks` `[n_rows]` int; `d_list_sizes_new/old` `[n_rows]` int2

### 6–9. ACE model ([`build_ace`](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L829))
Tunables ([cagra_build.cuh:822](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L822)): `usable_cpu_memory_fraction=0.8`, `usable_gpu_memory_fraction=0.8`, `imbalance_factor=3.0`, `vector_expansion_factor=2.0`.

`sub_partition_size = imbalance_factor · vector_expansion_factor · (dataset_size / n_partitions)`

| Field ([`ace_memory_requirements`](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L810)) | Formula |
|---|---|
| `partition_labels_size` | `2 · dataset_size · sizeof(IdxT)` |
| `id_mapping_size` | `2 · dataset_size · sizeof(IdxT)` |
| `sub_dataset_size` | `sub_partition_size · dataset_dim · sizeof(T)` |
| `sub_graph_size` | `sub_partition_size · (intermediate+graph_degree) · sizeof(IdxT)` |
| `cagra_graph_size` | `dataset_size · graph_degree · sizeof(IdxT)` |
| `total_size` | sum + `optimize_workspace_size` host + `2e9` |

- [`ace_check_use_disk_mode`](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L829): `host_limited = usable_cpu·avail_host < total_size`; `gpu_required = max(sub_dataset_size, opt_dev_ws) + 2e9`; `gpu_limited = usable_gpu·avail_gpu < gpu_required`. `use_disk = use_disk || host_limited || gpu_limited`.
- [`ace_validate_disk_mode_partitions`](../../cpp/src/neighbors/detail/cagra/cagra_build.cuh#L943): recomputes disk-mode host/GPU requirements and increases `n_partitions` (via `*_suggested_partitions`) until per-partition footprint fits.

### 10–12. HNSW decisions
| Decision | Logic |
|---|---|
| in‑memory vs ACE ([`hnsw::build`](../../cpp/src/neighbors/detail/hnsw.hpp#L1493)) | [`cagra_build_mem_usage`](../../cpp/src/neighbors/detail/cagra/cagra_helpers.cpp#L112) vs [`get_available_memory`](../../cpp/src/neighbors/detail/hnsw.hpp#L1175); if `!(req_host<avail_host && req_dev<avail_dev)` → `use_ace=true` |
| spill HNSW to disk ([`from_cagra`](../../cpp/src/neighbors/detail/hnsw.hpp#L1255)) | `required_host = n_rows · size_data_per_element · 1.125` vs `available_host` → spill if `>=` |
| [`get_available_memory`](../../cpp/src/neighbors/detail/hnsw.hpp#L1175) | host: `get_free_host_memory()` or `max_host_memory_gb`; device: `rmm::available_device_memory().second` or `max_gpu_memory_gb` |

# Open5M allocations
<img width="1650" height="6525" alt="openai_5M" src="https://github.com/user-attachments/assets/e6865c29-b44f-49cc-a7d7-5feaa73b6387" />
<img width="2100" height="1920" alt="openai_5M_timeline" src="https://github.com/user-attachments/assets/cf2322c2-e93b-4941-8373-b444025830da" />
