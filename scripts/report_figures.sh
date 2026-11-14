#!/usr/bin/env bash
# Figures for the report (reports/ataskaita/): the same plots as the stage reports, but
# without stage titles. Output: reports/ataskaita/img/<figure>/plots/*.png
# Run by `make report` before scripts/build_report.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
CSV=results/csv
IMG=reports/ataskaita/img
TLB="3547:TLB riba (n ≈ 3547)"

# upto <variant> <max n>: CSV files of the variant with n <= max n
upto() {
    local f n
    for f in "$CSV/$1"-[0-9]*.csv; do
        n=${f#"$CSV/$1"-}; n=${n%%x*}
        (( n <= $2 )) && echo "$f"
    done
}

fig() {  # fig <name> <plot.py options...>
    local name=$1; shift
    "$PY" scripts/plot.py --output "$IMG/$name" "$@" > /dev/null
    echo "$IMG/$name"
}

# Bendroji dalis (0_ivadas)
fig tlb    --input $CSV/seq-base-*.csv $CSV/openmp-base-*.csv $CSV/mpi-base-*.csv \
           --sizes 2000-4000 --workers 1 --vline "$TLB" --plots size
fig seqopt --input $CSV/seq-base-*.csv $CSV/seq-opt-*.csv --vline "$TLB" --plots size

# 1 dalis: OpenMP (each variant up to the size at which it was abandoned)
OMP="$(upto openmp-base 4000) $(upto openmp-opt 8000) $CSV/openmp-opt2-*.csv"
fig omp        --input $OMP --plots time speedup size
fig omp_amdahl --input $CSV/openmp-opt2-*.csv --e-ratio 0.15 --plots amdahl phases fraction karpflatt

# 2 dalis: MPI
fig mpi        --input $(upto mpi-base 4000) $CSV/mpi-opt-*.csv --plots time speedup size
fig mpi_amdahl --input $CSV/mpi-opt-*.csv --e-ratio 0.15 --plots amdahl phases karpflatt

# 3 dalis: OpenCL
fig ocl        --input $CSV/opencl-*.csv --plots size

# Bendros išvados
fig final      --input $CSV/seq-base-*.csv $CSV/seq-opt-*.csv $CSV/openmp-opt2-*.csv $CSV/mpi-opt-*.csv \
                       $CSV/opencl-opt-regblock-*.csv --plots size
fig final_f2   --input $CSV --plots fill2
"$PY" scripts/plot_final.py reports/ataskaita/img/final > /dev/null
