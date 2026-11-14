#!/usr/bin/env bash
# Benchmark sweep: every built variant x matrix sizes x worker counts.
#
# Output: results/csv/<config>.csv, one file per configuration (header + one row per
# repeat). Every sweep appends its rows at the end; existing rows are kept (read by plot.py).
# <config>:
#   seq / openmp / mpi:  <binary>-<n>x<k>x<p>-C<workers>   e.g. openmp-base-2000x2000x2000-C8
#   opencl:              <binary>-<kernel>-<n>x<k>x<p>      e.g. opencl-base-naive-2000x2000x2000
#                        (every kernel in src/opencl/<binary suffix>/*.cl is run)
#
# Console: the known part of a row (configuration) is printed before the run,
# the measured part (times, repeat) is appended to the same line after it.
#
# Override via environment, e.g.:
#   SIZES="500 1000" THREADS="1 2 4 8" REPEATS=5 VARIANTS="openmp-base mpi-base" scripts/bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"250 500 1000 2000 4000"}
THREADS=${THREADS:-"1 2 4 8 12"}
REPEATS=${REPEATS:-3}
SEED=${SEED:-42}
VARIANTS=${VARIANTS:-$(cd build && ls -p | grep -v /)}   # programs only, not build/tools/

HEADER="tech,variant,n,k,p,workers,fill1,fill2,mul,total,repeat"
OUT_DIR=results/csv
mkdir -p "$OUT_DIR"

# run_config <config-name> <expected-prefix> <command...>
#   Runs the command REPEATS times. Each output must be one CSV line of 10 fields
#   starting with <expected-prefix>; anything else is reported and not saved.
#   Valid rows are appended to $OUT_DIR/<config-name>.csv (header only for a new file).
run_config() {
    local name=$1 prefix=$2; shift 2
    local out=$OUT_DIR/$name.csv line r
    for r in $(seq 1 "$REPEATS"); do
        printf '%s' "$prefix"
        if ! line=$("$@" --seed "$SEED" --csv 2>&1); then
            printf 'FAILED (exit code): %s\n' "$*"
            continue
        fi
        if [[ $line != "$prefix"* || $(awk -F, '{print NF}' <<<"$line") != 10 || $line == *$'\n'* ]]; then
            printf 'FAILED (unexpected output): %s\n' "$*"
            printf '    | %s\n' "${line//$'\n'/$'\n    | '}"
            continue
        fi
        printf '%s,%s\n' "${line#"$prefix"}" "$r"
        [[ -s $out ]] || echo "$HEADER" > "$out"
        printf '%s,%s\n' "$line" "$r" >> "$out"
    done
}

echo "$HEADER"
for v in $VARIANTS; do
    bin=build/$v
    tech=${v%%-*}
    variant=${v#*-}
    for s in $SIZES; do
        dims=${s}x${s}x${s}
        sz=$s,$s,$s
        case $tech in
            seq)
                run_config "$v-$dims-C1" "$tech,$variant,$sz,1," "$bin" -s "$s" ;;
            openmp)
                for t in $THREADS; do
                    run_config "$v-$dims-C$t" "$tech,$variant,$sz,$t," "$bin" -s "$s" -t "$t"
                done ;;
            mpi)
                for t in $THREADS; do
                    run_config "$v-$dims-C$t" "$tech,$variant,$sz,$t," mpirun -np "$t" "$bin" -s "$s"
                done ;;
            opencl)
                # kernels of build/opencl-<X> are src/opencl/<X>/*.cl;
                # CSV variant = kernel name; workers (compute units) are known only after the run
                for cl in src/opencl/"$variant"/*.cl; do
                    k=$(basename "$cl" .cl)
                    run_config "$v-$k-$dims" "$tech,$k,$sz," "$bin" -s "$s" --kernel "$cl"
                done ;;
            *) echo "Skipping unknown variant: $v" ;;
        esac
    done
done

echo "Appended to: $OUT_DIR/*.csv"
