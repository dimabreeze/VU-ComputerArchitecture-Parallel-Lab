#!/usr/bin/env bash
# Stage reports: plots for every stage of the work, from results/csv/*.csv.
#
#   scripts/reports.sh            all stages
#   scripts/reports.sh 02 05      only stages whose directory starts with 02 / 05
#
# Each stage writes reports/<stage>/plots/*.png (plot.py --output reports/<stage>).
# reports/<stage>/README.md is written by hand and refers to these plots; it is not
# touched here. Existing files are kept, plots with the same names are overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
CSV=results/csv
TLB="3547:TLB riba (n ≈ 3547)"   # docs/tlb_cliff.md

# stage <dir> <title> <plot.py options...>
stage() {
    local dir=$1 title=$2 sel
    shift 2
    if [[ ${#ONLY[@]} -gt 0 ]]; then
        for sel in "${ONLY[@]}"; do [[ $dir == "$sel"* ]] && break; sel=; done
        [[ -n $sel ]] || return 0
    fi
    echo "== reports/$dir"
    "$PY" scripts/plot.py --output "reports/$dir" --title "$title" "$@"
}
ONLY=("$@")

# upto <variant> <max n>: CSV files of the variant with n <= max n. Used where the
# variant was abandoned at that size (larger sizes have single runs and add nothing).
upto() {
    local f n
    for f in "$CSV/$1"-[0-9]*.csv; do
        n=${f#"$CSV/$1"-}; n=${n%%x*}
        (( n <= $2 )) && echo "$f"
    done
}

stage 01-baseline "1 etapas · Pradiniai variantai" \
    --input $CSV/seq-base-*.csv $CSV/openmp-base-*.csv $CSV/mpi-base-*.csv $CSV/opencl-base-*.csv \
    --sizes 2000-4000 --plots time speedup size

stage 02-tlb-cliff "2 etapas · i-j-z ciklo TLB riba" \
    --input $CSV/seq-base-*.csv $CSV/openmp-base-*.csv $CSV/mpi-base-*.csv \
    --sizes 2000-4000 --workers 1 --vline "$TLB" --plots size

stage 03-seq-opt "3 etapas · Nuoseklus optimizavimas: seq-base → seq-opt" \
    --input $CSV/seq-base-*.csv $CSV/seq-opt-*.csv \
    --vline "$TLB" --plots size

stage 04-openmp "4 etapas · OpenMP: base → opt → opt2" \
    --input $(upto openmp-base 4000) $(upto openmp-opt 8000) $CSV/openmp-opt2-*.csv \
    --plots time speedup size

stage 05-mpi "5 etapas · MPI: base → opt" \
    --input $(upto mpi-base 4000) $CSV/mpi-opt-*.csv \
    --plots time speedup size

stage 06-opencl "6 etapas · OpenCL: naive → coalesced → tiled → regblock" \
    --input $CSV/opencl-*.csv \
    --plots size

# the same directory twice: the fill2 "bell" from all runs, Amdahl's law per variant.
# E_RATIO: E-core / P-core speed of openmp-opt2, measured with taskpolicy -b (docs/amdahl.md)
E_RATIO=0.15
stage 07-amdahl "7 etapas · Nuoseklioji dalis ir Amdalo dėsnis" \
    --input $CSV \
    --plots fill2
stage 07-amdahl "7 etapas · Nuoseklioji dalis ir Amdalo dėsnis" \
    --input $(upto openmp-base 4000) $(upto openmp-opt 8000) $CSV/openmp-opt2-*.csv $CSV/mpi-opt-*.csv \
    --e-ratio $E_RATIO --plots amdahl phases fraction karpflatt

stage 08-final "8 etapas · Geriausių variantų palyginimas" \
    --input $CSV/seq-base-*.csv $CSV/seq-opt-*.csv $CSV/openmp-opt2-*.csv $CSV/mpi-opt-*.csv \
            $CSV/opencl-opt-regblock-*.csv \
    --plots size

echo "Reports: reports/*/plots/"
