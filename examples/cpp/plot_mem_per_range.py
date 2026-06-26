#!/usr/bin/env python3
"""
plot_mem.py - Visualize peak allocation attributed to each NVTX range.

Reads a CSV produced by raft::memory_tracking_resources and, for each NVTX range
and memory source (host, device, workspace, large_workspace), reports the peak
memory the range is RESPONSIBLE for.

RESPONSIBLE-RANGE ATTRIBUTION
-----------------------------
Each event row carries three columns that make attribution exact:
  event_source  - which source allocated/freed (host, device, ...)
  event_bytes   - signed bytes for THIS event (+allocation / -free)
  alloc_range   - the FULL root->leaf range path responsible for the memory,
                  captured AT ALLOCATION TIME, formatted "name#id > name#id > ..."

Because alloc_range is captured when the memory is allocated, a free is charged
back to the range that ALLOCATED the memory -- not the range that happens to be
active when the free runs. So memory allocated under range A and freed under a
later range B is correctly accounted to A.

NESTING (inclusive)
-------------------
alloc_range is the whole stack, so a byte allocated inside an inner range is
charged to that inner range AND to every enclosing (outer) range. We maintain a
running live total per (source, range-instance) and track its peak.

The "#id" is a unique instance id: re-entering a range creates a new instance.
Instances are merged BY NAME into a distribution, so a range entered N times
contributes N peak samples:
  bar = mean, |---| = min..max whiskers, n = #instances

Allocations made outside any range are bucketed as "(no range)".

Usage:
    python plot_mem.py [path/to/stats.csv] [-o out.png] [--top N] [-f REGEX]
                       [--include-zeros]

Use -f/--filter to keep only ranges matching a regex, e.g.:
    python plot_mem.py -f build        # ranges containing "build"
    python plot_mem.py -f '^aaa'       # ranges starting with "aaa"
    python plot_mem.py -f '(?i)build'  # case-insensitive
"""

import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")  # no display needed; write a PNG file
import matplotlib.pyplot as plt
import numpy as np

# The four sources we visualize, in legend order.
SOURCES = ["host", "pinned", "workspace", "large_workspace", "managed", "device", ]

# Colors for the sources
COLORS = {
    "host": "#FF0000",            # red
    "pinned": "#FFA500",          # orange
    "workspace": "#009E73",       # green
    "large_workspace": "#008000", # dark green
    "managed": "#999999",         # grey
    "device": "#0000FF",          # blue
}


def human_bytes(n):
    """Format a byte count as a short human-readable string (1024-based)."""
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0


NO_RANGE = "(no range)"


def parse_range_path(path):
    """Parse an alloc_range string into [(name, instance_id), ...] root->leaf.

    Format is "name#id > name#id > ...". Splits on " > ", then separates the
    trailing "#<id>" with rpartition so range names containing '#' survive.
    A malformed token (no numeric id) is kept with its raw text as the instance
    key, so it still buckets consistently. Returns [] for an empty path.
    """
    path = (path or "").strip()
    if not path:
        return []
    nodes = []
    for tok in path.split(" > "):
        name, sep, idstr = tok.rpartition("#")
        if sep and idstr.isdigit():
            nodes.append((name, int(idstr)))
        else:
            nodes.append((tok, tok))
    return nodes


def load_values(csv_path, include_zeros=False):
    """
    Return (values, depths):
      values: OrderedDict range_name -> {source: [peak_inclusive_bytes per instance]}
      depths: range_name -> set of stack depths observed for that name

    Responsible-range, inclusive attribution (see module docstring): for each
    event we add its signed event_bytes to a running live total for EVERY range
    on its alloc_range path (inner and all enclosing ranges), keyed by
    (event_source, range-instance), and track the peak of that live total.

    Each range instance (a unique "#id") yields one peak sample; instances are
    merged by name into a distribution, so a range entered N times gives N
    samples. Allocations outside any range bucket as "(no range)".
    """
    required = ["event_source", "event_bytes", "alloc_range"]
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in required:
            if col not in fieldnames:
                sys.exit(f"ERROR: CSV is missing required column '{col}'.\n"
                         f"Found columns: {fieldnames}")
        rows_data = list(reader)

    live = {}        # (source, inst_key) -> running inclusive live bytes
    peak = {}        # (source, inst_key) -> peak inclusive live bytes
    inst_name = {}   # inst_key -> range name
    inst_depth = {}  # inst_key -> stack depth (1-based position on the path)
    first_seen = {}  # name -> order index of first appearance (timeline order)
    order = 0

    for row in rows_data:
        src = (row.get("event_source") or "").strip()
        if not src:
            continue
        try:
            delta = int(row["event_bytes"])
        except (ValueError, TypeError):
            continue

        nodes = parse_range_path(row.get("alloc_range")) or [(NO_RANGE, NO_RANGE)]
        for depth, (name, iid) in enumerate(nodes, start=1):
            if name not in first_seen:
                first_seen[name] = order
                order += 1
            inst_name[iid] = name
            inst_depth[iid] = depth
            key = (src, iid)
            live[key] = live.get(key, 0) + delta
            if live[key] > peak.get(key, 0):
                peak[key] = live[key]

    # Build the result skeleton in timeline order (first appearance of each name).
    values = OrderedDict()
    depths = {}
    for name in sorted(first_seen, key=first_seen.get):
        values[name] = {s: [] for s in SOURCES}
        depths[name] = set()

    # One peak sample per (source, instance), merged under the instance's name.
    for (src, iid), pk in peak.items():
        name = inst_name[iid]
        if src not in values[name]:
            # event_source not in the known SOURCES list; ignore for plotting.
            continue
        if pk > 0 or include_zeros:
            values[name][src].append(pk)
            depths[name].add(inst_depth[iid])

    return values, depths


def filter_ranges(values, pattern):
    """Keep only ranges whose label matches the regex (re.search). Order kept."""
    if not pattern:
        return values
    try:
        rx = re.compile(pattern)
    except re.error as e:
        sys.exit(f"ERROR: invalid regex {pattern!r}: {e}")
    kept = OrderedDict((lbl, v) for lbl, v in values.items() if rx.search(lbl))
    if not kept:
        sys.exit(f"ERROR: no NVTX range matched regex {pattern!r}.")
    return kept


def _max_peak(per_source):
    """Largest peak_delta across all sources for one range (0 if no data)."""
    m = 0
    for lst in per_source.values():
        if lst:
            m = max(m, max(lst))
    return m


def rank(values, top):
    """Return list of (label, per_source) sorted by max peak_delta desc; keep top-N."""
    items = sorted(values.items(), key=lambda kv: _max_peak(kv[1]), reverse=True)
    if top and top > 0:
        items = items[:top]
    return items


def print_table(items, total_ranges):
    """Print per-range, per-source stats."""
    print(f"Showing {len(items)} of {total_ranges} ranges "
          f"(top-N by max peak delta, listed in timeline order)\n")
    for label, per_source in items:
        print(label)
        for s in SOURCES:
            v = per_source[s]
            if not v:
                continue
            a = np.asarray(v, dtype=float)
            print(f"    {s:16} n={len(v):<4} mean={human_bytes(a.mean())}"
                  f"  std={human_bytes(a.std())}"
                  f"  min={human_bytes(a.min())}  max={human_bytes(a.max())}")


def plot(items, depths, out_path, title):
    # `items` arrive in timeline order (earliest first). Reverse for bottom-up y axis.
    items = list(reversed(items))
    labels = []
    for lbl, _ in items:
        ds = depths.get(lbl) or set()
        if ds:
            dmin, dmax = min(ds), max(ds)
            dtxt = f"depth {dmin}" if dmin == dmax else f"depth {dmin}-{dmax}"
            labels.append(f"{lbl}\n({dtxt})")
        else:
            labels.append(lbl)
    n_ranges = len(labels)
    n_sources = len(SOURCES)

    y = np.arange(n_ranges)
    slot = 0.8 / n_sources

    fig_h = max(4.0, 0.70 * n_ranges * n_sources / 2.0 + 1.5)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    def label(x, p, text, ha, dx, dy=0, color="0.15"):
        ax.annotate(text, xy=(x, p), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=("bottom" if dy else "center"), fontsize=5.5, color=color)

    for i, s in enumerate(SOURCES):
        offset = (i - (n_sources - 1) / 2.0) * slot

        rows = []
        for yi, (_, per_source) in enumerate(items):
            v = per_source[s]
            if not v:
                continue
            a = np.asarray(v, dtype=float)
            rows.append((yi + offset, a.mean(), a.std(), a.min(), a.max(), len(v)))
        if not rows:
            continue

        pos  = np.array([r[0] for r in rows])
        mean = np.array([r[1] for r in rows])
        lo   = np.array([r[3] for r in rows])
        hi   = np.array([r[4] for r in rows])
        cnt  = [r[5] for r in rows]

        ax.barh(pos, mean, slot * 0.9, color=COLORS[s], alpha=0.30, label=s, zorder=1)
        ax.errorbar(mean, pos, xerr=[mean - lo, hi - mean], fmt="none",
                    ecolor=COLORS[s], elinewidth=1.0, capsize=3, zorder=2)
        ax.plot(mean, pos, "D", ms=3.0, color=COLORS[s], zorder=4)

        for p, m, l, h, c in zip(pos, mean, lo, hi, cnt):
            if h <= l * 1.0001:
                label(m, p, f"{human_bytes(m)} (n={c})", "left", 5)
            else:
                label(l, p, human_bytes(l), "right", -3, color=COLORS[s])
                label(h, p, human_bytes(h), "left", 3, color=COLORS[s])
                label(m, p, f"{human_bytes(m)} (n={c})", "center", 0, dy=4)

    ax.set_xscale("log")
    ax.set_xlabel("Peak memory attributed to range (inclusive, log scale)")
    ax.set_ylabel("NVTX range")
    ax.set_title(f"{title}\nbar = mean,  |---| = min..max,  n = #occurrences", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.legend(title="source", loc="lower right")
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to: {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "hnsw_build_stats.csv")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=default_csv,
                   help="path to the stats CSV (default: %(default)s)")
    p.add_argument("-o", "--out", default=None,
                   help="output PNG path (default: <csv basename>_<kind>.png)")
    p.add_argument("--top", type=int, default=20,
                   help="plot only the top-N ranges by max peak delta (0 = all; default: 20)")
    p.add_argument("-f", "--filter", default=None, metavar="REGEX",
                   help="keep only ranges matching this regex (re.search). "
                        "Examples: 'build', '^aaa', '(?i)build' for case-insensitive.")
    p.add_argument("--include-zeros", action="store_true",
                   help="keep occurrences where a source's peak delta is 0 (default: drop them).")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    out_path = args.out or os.path.splitext(args.csv)[0] + ".png"

    values, depths = load_values(args.csv, include_zeros=args.include_zeros)
    if not values:
        sys.exit("ERROR: no data rows found in CSV.")

    values = filter_ranges(values, args.filter)
    timeline = {lbl: i for i, lbl in enumerate(values)}
    items = rank(values, args.top)
    items.sort(key=lambda kv: timeline[kv[0]])
    print_table(items, total_ranges=len(values))
    plot(items, depths, out_path,
         title=f"Peak memory attributed per NVTX range  -  {os.path.basename(args.csv)}")


if __name__ == "__main__":
    main()
