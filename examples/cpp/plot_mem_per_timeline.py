#!/usr/bin/env python3
"""
plot_mem_per_timeline.py - Visualize memory allocations over a timeline.

Reads a CSV produced by raft::memory_tracking_resources and draws one panel
per memory source plus a bottom NVTX panel, all sharing the same time axis:

  Per-source panels (filled area) — one each for:
      device, workspace, large_workspace, managed, host, pinned
  BOTTOM — NVTX ranges as horizontal bars, one horizontal lane per depth
            level (depth 1 at top, deeper levels below).  When multiple
            occurrences at the same depth overlap in time they are stacked
            into sub-rows within that depth lane.

The x-axis shows elapsed seconds from the first recorded timestamp.

Usage:
    python plot_mem_per_timeline.py [path/to/stats.csv] [-o out.png]
                                    [-f REGEX]

Use -f/--filter to keep only ranges matching a regex in the bottom panel.
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Sources visualized in the top (device-side) panel.
DEVICE_SOURCES = ["device", "workspace", "large_workspace", "managed"]

# Sources visualized in the middle (host-side) panel.
HOST_SOURCES = ["host", "pinned"]

# Colors for the sources
COLORS = {
    "host": "#FF0000",            # red
    "pinned": "#FFA500",          # orange
    "workspace": "#009E73",       # green
    "large_workspace": "#008000", # dark green
    "managed": "#999999",         # grey
    "device": "#0000FF",          # blue
}

# One distinct color per NVTX depth level (for the bottom panel).
DEPTH_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]


def human_bytes(n):
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0


def bytes_formatter():
    """Return a matplotlib FuncFormatter that shows byte counts as human-readable."""
    def fmt(x, _pos):
        if x == 0:
            return "0"
        return human_bytes(x)
    return ticker.FuncFormatter(fmt)


def load_csv(csv_path):
    """Return (fieldnames, rows) where rows is a list of dicts."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def validate_columns(fieldnames, sources):
    for s in sources:
        if f"{s}_current" not in fieldnames:
            sys.exit(
                f"ERROR: CSV is missing '{s}_current' column.\n"
                f"Found columns: {fieldnames}"
            )
    for col in ("timestamp_us", "nvtx_range", "nvtx_depth"):
        if col not in fieldnames:
            sys.exit(f"ERROR: CSV is missing required column '{col}'.\n"
                     f"Found columns: {fieldnames}")


def extract_timeline(rows, sources):
    """Return (t, data) where t is a float array of elapsed seconds and
    data is a dict source -> float array of current bytes."""
    t0 = None
    ts, vals = [], {s: [] for s in sources}
    for row in rows:
        try:
            us = int(row["timestamp_us"])
        except (ValueError, TypeError):
            continue
        if t0 is None:
            t0 = us
        ts.append((us - t0) / 1e6)
        for s in sources:
            try:
                vals[s].append(int(row[f"{s}_current"]))
            except (ValueError, TypeError):
                vals[s].append(0)
    t = np.array(ts, dtype=float)
    data = {s: np.array(vals[s], dtype=float) for s in sources}
    return t, data


def extract_spans(rows, filter_pattern=None):
    """Reconstruct NVTX range spans using a stack so that outer ranges span
    the full duration including any nested inner ranges.

    Returns a list of dicts: {label, depth, t_start, t_end} (times in seconds).

    Example: outer(depth=1) → inner(depth=2) → outer(depth=1) yields
      outer: [entry … final_exit]   (one continuous bar)
      inner: [entry … exit]         (nested bar inside outer)
    """
    if not rows:
        return []

    try:
        t0 = int(rows[0]["timestamp_us"])
    except (ValueError, TypeError):
        t0 = 0

    rx = None
    if filter_pattern:
        try:
            rx = re.compile(filter_pattern)
        except re.error as e:
            sys.exit(f"ERROR: invalid regex {filter_pattern!r}: {e}")

    def sec(us):
        return (us - t0) / 1e6

    def emit(label, depth, start_us, end_us):
        if not label:
            return
        if rx and not rx.search(label):
            return
        spans.append({"label": label, "depth": depth,
                      "t_start": sec(start_us), "t_end": sec(end_us)})

    spans = []
    # stack[i] = (name, start_us) for depth i+1
    stack = []
    prev_us = t0

    for row in rows:
        try:
            us = int(row["timestamp_us"])
        except (ValueError, TypeError):
            continue
        label = row.get("nvtx_range", "").strip()
        try:
            depth = int(row["nvtx_depth"])
        except (ValueError, TypeError):
            depth = 0

        cur_depth = len(stack)

        if depth > cur_depth:
            # Entering one or more deeper levels.
            stack.append((label, us))
        elif depth == cur_depth:
            if depth == 0:
                pass  # still in the "no range" zone
            elif stack[-1][0] != label:
                # Same depth, different range: close current, open new.
                name, start = stack.pop()
                emit(name, cur_depth, start, us)
                stack.append((label, us))
        else:
            # Exiting to a shallower depth: close all deeper ranges.
            while len(stack) > depth:
                name, start = stack.pop()
                d = len(stack) + 1
                emit(name, d, start, us)
            # If still inside a range at the new depth, check whether it
            # changed (e.g. a sibling range starts immediately after).
            if depth > 0 and stack and stack[-1][0] != label:
                name, start = stack.pop()
                emit(name, depth, start, us)
                stack.append((label, us))

        prev_us = us

    # Close any ranges still open at the end of the file.
    for i, (name, start) in enumerate(stack):
        emit(name, i + 1, start, prev_us)

    return spans


def layout_spans(spans):
    """Assign a sub-row index to each span to avoid overlap within a depth.
    Returns list of dicts with added 'sub_row' key.
    Also returns max_sub_row per depth: dict depth -> int.
    """
    from collections import defaultdict

    # Group by depth, process in time order.
    by_depth = defaultdict(list)
    for sp in spans:
        by_depth[sp["depth"]].append(sp)

    max_sub = {}
    result = []
    for depth, group in by_depth.items():
        group.sort(key=lambda s: s["t_start"])
        # Greedy lane assignment: track the earliest end time of each sub-row.
        sub_row_ends = []
        for sp in group:
            placed = False
            for sub, end in enumerate(sub_row_ends):
                if sp["t_start"] >= end:
                    sub_row_ends[sub] = sp["t_end"]
                    sp = dict(sp, sub_row=sub)
                    placed = True
                    break
            if not placed:
                sp = dict(sp, sub_row=len(sub_row_ends))
                sub_row_ends.append(sp["t_end"])
            result.append(sp)
        max_sub[depth] = len(sub_row_ends) - 1

    return result, max_sub


def plot(t, data, spans, out_path, csv_name):
    # Only include sources that have any non-zero data.
    active_sources = [s for s in DEVICE_SOURCES + HOST_SOURCES
                      if s in data and data[s].any()]

    # Compute NVTX depth layout.
    annotated_spans, max_sub = layout_spans(spans)
    depths_present = sorted(set(sp["depth"] for sp in annotated_spans if sp["depth"] > 0))

    n_source_panels = len(active_sources)
    bottom_h = max(1.5, sum(max_sub.get(d, 0) + 1 for d in depths_present) * 0.6 + 1.0)
    source_panel_h = 2.0  # height per source panel

    fig_h = n_source_panels * source_panel_h + bottom_h + 0.8
    height_ratios = [source_panel_h] * n_source_panels + [bottom_h]

    fig, axes = plt.subplots(
        n_source_panels + 1, 1,
        figsize=(14, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    if n_source_panels + 1 == 1:
        axes = [axes]

    source_axes = axes[:n_source_panels]
    ax_nvtx = axes[n_source_panels]

    # --- Per-source filled-area panels ---
    for i, (s, ax) in enumerate(zip(active_sources, source_axes)):
        color = COLORS[s]
        ax.fill_between(t, data[s], color=color, alpha=0.4, linewidth=0, step="post")
        ax.plot(t, data[s], color=color, linewidth=0.8, drawstyle="steps-post")
        ax.set_ylabel(s, fontsize=9)
        ax.yaxis.set_major_formatter(bytes_formatter())
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.set_title(f"Memory timeline  —  {csv_name}", fontsize=11)

    # --- Bottom: NVTX ranges ---
    y_cursor = 0.0
    depth_y_base = {}
    for d in depths_present:
        depth_y_base[d] = y_cursor
        n_sub = max_sub.get(d, 0) + 1
        y_cursor += n_sub + 0.4

    bar_h = 0.75

    for d in depths_present:
        n_sub = max_sub.get(d, 0) + 1
        mid_y = depth_y_base[d] + (n_sub - 1) / 2.0
        ax_nvtx.text(
            -0.002, mid_y, f"depth {d}",
            transform=ax_nvtx.get_yaxis_transform(),
            ha="right", va="center", fontsize=7, color="0.4",
        )

    color_cache = {}

    def range_color(label):
        if label not in color_cache:
            idx = len(color_cache) % len(DEPTH_COLORS)
            color_cache[label] = DEPTH_COLORS[idx]
        return color_cache[label]

    t_span = (t[-1] - t[0]) if t.size > 1 else 1.0
    for sp in annotated_spans:
        d = sp["depth"]
        if d == 0:
            continue
        sub = sp["sub_row"]
        y = depth_y_base[d] + sub
        w = max(sp["t_end"] - sp["t_start"], 0.0)
        color = range_color(sp["label"])
        ax_nvtx.barh(y, w, left=sp["t_start"], height=bar_h,
                     color=color, alpha=0.75, edgecolor="none")
        if w > t_span * 0.01:
            ax_nvtx.text(
                sp["t_start"] + w / 2, y, sp["label"],
                ha="center", va="center", fontsize=5.5, clip_on=True,
                color="white", fontweight="bold",
            )

    total_y = y_cursor if y_cursor > 0 else 1.0
    ax_nvtx.set_ylim(-0.5, total_y - 0.4 + 0.5)
    ax_nvtx.set_yticks([])
    ax_nvtx.set_xlabel("Elapsed time (s)")
    ax_nvtx.set_ylabel("NVTX ranges")
    ax_nvtx.grid(axis="x", linestyle=":", alpha=0.4)

    if t.size:
        for ax in axes:
            ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "hnsw_build_stats.csv")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=default_csv,
                   help="path to the stats CSV (default: %(default)s)")
    p.add_argument("-o", "--out", default=None,
                   help="output PNG path (default: <csv basename>_timeline.png)")
    p.add_argument("-f", "--filter", default=None, metavar="REGEX",
                   help="show only NVTX ranges matching this regex in the bottom panel.")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    out_path = args.out or os.path.splitext(args.csv)[0] + "_timeline.png"

    fieldnames, rows = load_csv(args.csv)
    validate_columns(fieldnames, DEVICE_SOURCES + HOST_SOURCES)

    all_sources = DEVICE_SOURCES + HOST_SOURCES
    t, data = extract_timeline(rows, all_sources)

    spans = extract_spans(rows, filter_pattern=args.filter)

    plot(t, data, spans, out_path, csv_name=os.path.basename(args.csv))


if __name__ == "__main__":
    main()
