#!/usr/bin/env python3
"""Report peak memory per source from a memory_tracking_resources CSV.

The CSV (produced by raft::memory_tracking_resources) has one row per memory
event, with a `<pool>_current` column giving the live bytes of each pool at that
timestamp. We report the per-source peak (each pool's individual maximum) as
well as the combined device and host totals.

Sources:
  host           = raft::make_host_* (malloc)
  pinned         = pinned host memory
  workspace      = get_workspace_resource
  large_workspace= get_large_workspace_resource
  managed        = CUDA Unified Memory (cudaMallocManaged)
  device         = default device allocator

Combined totals:
  device_total = device + workspace + large_workspace + managed
  host_total   = host + pinned

Usage:
    ./get_max.py openai_5M_ace.csv
"""

import argparse
import csv
import sys

ALL_POOLS = ["host", "pinned", "workspace", "large_workspace", "managed", "device"]
DEVICE_POOLS = ["device", "workspace", "large_workspace", "managed"]
HOST_POOLS = ["host", "pinned"]


def to_gib(n_bytes: int) -> float:
    return n_bytes / float(1 << 30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="memory tracking CSV (e.g. openai_5M_ace.csv)")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"error: {args.csv_path} is empty", file=sys.stderr)
            return 1

        col = {name: i for i, name in enumerate(header)}

        def idx(pool: str) -> int:
            key = f"{pool}_current"
            if key not in col:
                print(f"error: column '{key}' not found in {args.csv_path}", file=sys.stderr)
                sys.exit(1)
            return col[key]

        ts_idx      = col.get("timestamp_us")
        pool_idx    = {p: idx(p) for p in ALL_POOLS}
        device_idx  = [pool_idx[p] for p in DEVICE_POOLS]
        host_idx    = [pool_idx[p] for p in HOST_POOLS]

        max_per_pool  = {p: 0 for p in ALL_POOLS}
        max_device    = max_host = 0
        max_device_ts = max_host_ts = None

        for row in reader:
            if not row:
                continue
            for p in ALL_POOLS:
                v = int(row[pool_idx[p]])
                if v > max_per_pool[p]:
                    max_per_pool[p] = v

            device_sum = sum(int(row[i]) for i in device_idx)
            host_sum   = sum(int(row[i]) for i in host_idx)
            ts = row[ts_idx] if ts_idx is not None else None
            if device_sum > max_device:
                max_device, max_device_ts = device_sum, ts
            if host_sum > max_host:
                max_host, max_host_ts = host_sum, ts

    print(f"file: {args.csv_path}")
    print()
    print("peak per source:")
    width = max(len(p) for p in ALL_POOLS)
    for p in ALL_POOLS:
        v = max_per_pool[p]
        print(f"  {p:<{width}} : {to_gib(v):8.3f} GiB  ({v} bytes)")
    print()
    print("peak combined totals:")
    print(f"  {'device_total':<{width}} : {to_gib(max_device):8.3f} GiB  ({max_device} bytes)"
          f"  [device + workspace + large_workspace + managed]"
          + (f"  @ t={max_device_ts}us" if max_device_ts is not None else ""))
    print(f"  {'host_total':<{width}} : {to_gib(max_host):8.3f} GiB  ({max_host} bytes)"
          f"  [host + pinned]"
          + (f"  @ t={max_host_ts}us" if max_host_ts is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
