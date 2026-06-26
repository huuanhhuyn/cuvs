#!/usr/bin/env python3
"""
get_range.py - Return all active NVTX ranges at a given elapsed time.

Usage:
    python get_range.py <csv_file> <elapsed_seconds>
    python get_range.py <csv_file> <elapsed_seconds> --us   # time in microseconds

The elapsed time is relative to the first timestamp in the CSV.
Finds the last row whose timestamp <= the given time and prints all active ranges.
"""

import csv
import sys
import re


def parse_alloc_range(alloc_range: str) -> list[str]:
    """Parse 'A#1 > B#2 > C#3' into ['A', 'B', 'C']."""
    if not alloc_range:
        return []
    parts = alloc_range.split(" > ")
    return [re.sub(r"#\d+$", "", p.strip()) for p in parts]


def get_range(csv_file: str, elapsed_us: float) -> list[str]:
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    t0 = int(rows[0]["timestamp_us"])
    target = t0 + elapsed_us

    # Find the last row with timestamp <= target
    matching_row = None
    for row in rows:
        if int(row["timestamp_us"]) <= target:
            matching_row = row
        else:
            break

    if matching_row is None:
        return []

    return parse_alloc_range(matching_row["alloc_range"])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    csv_file = sys.argv[1]
    time_val = float(sys.argv[2])
    use_us = "--us" in sys.argv

    elapsed_us = time_val if use_us else time_val * 1_000_000

    ranges = get_range(csv_file, elapsed_us)

    if not ranges:
        print("No active ranges at that time.")
    else:
        print(f"Active NVTX ranges at elapsed {'%.0f us' % elapsed_us if use_us else '%.3f s' % (elapsed_us / 1e6)}:")
        for i, r in enumerate(ranges):
            indent = "  " * i
            marker = "└─ " if i > 0 else "   "
            print(f"{indent}{marker}{r}")


if __name__ == "__main__":
    main()
