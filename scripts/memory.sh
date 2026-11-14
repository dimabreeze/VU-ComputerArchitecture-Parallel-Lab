#!/usr/bin/env bash
# Memory microbenchmarks (build/tools/memlat) -> results/memory/*.csv -> docs/img/memory/*.png
# Run on an otherwise idle machine; takes about 3 minutes. See docs/memory.md.
#
# Every result is first written to a temporary file outside the project and then moved
# in one step: the project lives in OneDrive, which loses lines of a file that a program
# keeps open while writing to it (observed: 13 of 26 lines, exit code 0).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/memory
mkdir -p "$OUT"
tmp=$(mktemp)

run() {  # run <name> <repeats>: memlat <name>, repeated, into $OUT/<name>.csv
    local name=$1 reps=$2 i
    : > "$tmp"
    for ((i = 1; i <= reps; i++)); do build/tools/memlat "$name" >> "$tmp"; done
    mv "$tmp" "$OUT/$name.csv"
    tmp=$(mktemp)
    echo "$OUT/$name.csv: $(grep -vc '^mode' "$OUT/$name.csv") points"
}

run lat 1   # stable: one run
run tlb 3   # noisy above ~6000 pages: median of 3 in the plot
run bw 3    # thread placement varies: best of 3 in the plot

.venv/bin/python scripts/plot_memory.py
