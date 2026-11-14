#!/usr/bin/env python3
"""Plots for the report from bench CSVs.

Usage: scripts/plot.py [--input PATH ...] [--output DIR] [options]
  --input   CSV directories and/or files (default: results/csv). For a directory all
            *.csv in it are used, so several inputs or a shell glob select a subset:
            --input results/csv/seq-base-*.csv results/csv/seq-opt-*.csv
  --output  report directory (default: results). Plots go to <output>/plots/.
            Missing directories are created; existing content is kept, only files
            with the same names are overwritten.
  --sizes   keep only these matrix sizes n, e.g. 2000-4000 or 2000,4000,8000-16000
  --workers keep only these thread / process counts, same syntax, e.g. 1 or 1-8
  --title   stage title, shown as the first line of every plot title
  --vline   vertical marker on the size plots, X[:label], e.g. "3547:TLB limit";
            may be repeated
  --plots   which plots to draw (default: all):
            time speedup size fill2 amdahl phases fraction karpflatt
  --e-ratio speed of an efficiency core relative to a performance core (e.g. 0.15);
            adds <variant>-amdahl-hetero.png (Amdahl's law for 8 P + 4 E cores)
  --p-cores number of performance cores (default: 8)
Rows from all files are merged; repeats of the same (variant, size, workers)
are reduced to their median. Malformed rows are skipped with a warning.

Plots:
  <variant>-time.png, -time-linear.png  time vs workers, one line per size (openmp, mpi)
  <variant>-speedup.png                 speedup vs 1 worker of the same variant, one line
                                        per size that has a 1-worker run (openmp, mpi)
  size.png                              time vs n, log-log axes (best worker count): a power
                                        law is a straight line; dashed reference ∝ n³
  size-linear.png                       time vs the number of multiply-adds n·k·p, linear
                                        axes: a straight line for a kernel whose speed does
                                        not depend on n; memory effects show up as kinks
  <variant>-amdahl.png                  measured speedup (solid) and Amdahl's law (dashed)
                                        for every size, f measured as fill2 / total on 1 worker
  <variant>-amdahl-hetero.png           the same with N_eff = min(N, P) + r * max(N - P, 0)
  <variant>-phases.png                  time of every phase (fill1, fill2, mul) vs workers,
                                        one panel per size: the sequential fill2 stays
  amdahl-fraction.png                   serial fraction f vs n, Amdahl limit 1/f vs the
                                        best measured speedup (Gustafson)
  <variant>-karpflatt.png               Karp-Flatt metric e = (1/S - 1/N) / (1 - 1/N), the
                                        serial fraction implied by the measured speedup
  fill2-distribution.png                distribution ("bell") of the sequential fill
                                        time of matrix B for every size (>= 10 runs)
  fill2-elements.png                    median fill time of matrix B vs its number of
                                        elements k·p, with a fit through the origin
All benchmark runs are square (n = k = p), so n·k·p = n³ and k·p = n².
"""
import argparse
import csv
import locale
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# The default palette has only 10 colors, so series 11+ repeated colors of series 1+.
# Style cycle: 10 colors with solid lines and circles, then the same colors with
# dashed lines and squares, then dotted with triangles: 30 distinguishable series.
plt.rcParams["axes.prop_cycle"] = (
    cycler(linestyle=["-", "--", ":"]) + cycler(marker=["o", "s", "^"])
) * cycler(color=plt.get_cmap("tab10").colors)


# Plot labels are in Lithuanian; numbers use the decimal comma (lt_LT locale for the
# axis ticks, dec() for numbers inside labels).
try:
    locale.setlocale(locale.LC_NUMERIC, "lt_LT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass


def dec(x, fmt="g"):
    """number with a decimal comma: dec(3.14, '.1f') -> '3,1'"""
    return format(x, fmt).replace(".", ",")


SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(x):
    """8e9 -> '8·10⁹', 1.25e11 -> '1,25·10¹¹'"""
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    mant = "" if abs(m - 1) < 1e-9 else dec(round(m, 2)) + "·"
    return f"{mant}10{str(e).translate(SUPERSCRIPT)}"


def spread_ticks(values, min_gap):
    """subset of sorted values, neighbours at least min_gap apart (keeps the last one)"""
    out = []
    for v in sorted(values):
        if out and v - out[-1] < min_gap:
            continue
        out.append(v)
    if values and out[-1] != max(values):
        out[-1] = max(values)
    return out


def spread_log(values, ratio=1.3):
    """subset of sorted values for a log axis: each at least ratio times the previous"""
    out = []
    for v in sorted(values):
        if not out or v >= ratio * out[-1]:
            out.append(v)
    return out


def plain_ticks(axis):
    """plain numbers (0,1, 1, 10) instead of 10^x on a log axis"""
    axis.set_major_formatter(FuncFormatter(lambda y, _: dec(y)))
    axis.set_minor_formatter(NullFormatter())


# Time plots are drawn twice: log Y axis (default names) and linear Y axis (*-linear.png)
SCALES = (("log", ""), ("linear", "-linear"))
PLOTS = ("time", "speedup", "size", "fill2", "amdahl", "phases", "fraction", "karpflatt")

# Variants whose fill2 is exactly "one CPU thread fills matrix B". Excluded: mpi-base
# (fill2 also contains MPI_Bcast of B), so its fill2 is not the same quantity.
FILL2_PURE = ("seq-", "openmp-", "opencl-", "mpi-opt")


def legend_outside(ax, **kw):
    """Legend to the right of the plot area, so it never covers the curves.
    Figures are saved with bbox_inches="tight", which widens the image to fit it."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, **kw)


def int_set(spec):
    """'2000-4000,8000' -> predicate n -> bool (ranges are inclusive)."""
    parts = []
    for item in spec.split(","):
        lo, _, hi = item.strip().partition("-")
        parts.append((int(lo), int(hi or lo)))
    return lambda n: any(lo <= n <= hi for lo, hi in parts)


def vline(spec):
    """'3547:TLB limit' -> (3547.0, 'TLB limit')"""
    x, _, label = spec.partition(":")
    return float(x), label or None


def load(paths, keep_size=None, keep_workers=None):
    """Returns (medians, fill2 samples):
    medians: key (tech-variant, size, workers) -> median of repeats for each phase
    fill2:   size -> every individual fill2 value of the FILL2_PURE variants
    Rows rejected by keep_size / keep_workers are ignored."""
    runs = defaultdict(lambda: defaultdict(list))
    fill2 = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                try:
                    key = (f"{row['tech']}-{row['variant']}", int(row["n"]), int(row["workers"]))
                    vals = {col: float(row[col]) for col in ("fill1", "fill2", "mul", "total")}
                except (KeyError, TypeError, ValueError):
                    print(f"Skipping malformed row {path}:{line_no}", file=sys.stderr)
                    continue
                if (keep_size and not keep_size(key[1])) or (keep_workers and not keep_workers(key[2])):
                    continue
                for col, val in vals.items():
                    runs[key][col].append(val)
                if key[0].startswith(FILL2_PURE):
                    fill2[key[1]].append(vals["fill2"])
    medians = {k: {c: statistics.median(v) for c, v in cols.items()} for k, cols in runs.items()}
    return medians, fill2


def plot_fill2_distribution(fill2, path, title, min_samples=10):
    """One panel per matrix size: histogram of all fill2 runs (density) and the normal
    curve with the same mean and standard deviation. Sizes with fewer than
    min_samples runs are skipped (a histogram of 2 points says nothing)."""
    sizes = [n for n in sorted(fill2) if len(fill2[n]) >= min_samples]
    if not sizes:
        return False
    cols = min(3, len(sizes))
    rows = math.ceil(len(sizes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, n in zip(axes.flat, sizes):
        ms = [v * 1000 for v in fill2[n]]  # milliseconds
        mu, sigma = statistics.mean(ms), statistics.stdev(ms)
        ax.hist(ms, bins="auto", density=True, color="#9fb4cc", edgecolor="white", label="paleidimai")
        if sigma > 0:
            lo, hi = min(min(ms), mu - 4 * sigma), max(max(ms), mu + 4 * sigma)
            xs = [lo + (hi - lo) * i / 300 for i in range(301)]
            ys = [math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) for x in xs]
            ax.plot(xs, ys, color="#c0392b", linestyle="-", marker="", linewidth=2, label="normalusis skirstinys")
        ax.axvline(statistics.median(ms), color="#172033", linestyle="--", marker="", linewidth=1,
                   label="mediana")
        ax.set_title(f"n = {n}: μ = {dec(mu, '.1f')} ms, σ = {dec(sigma, '.1f')} ms, N = {len(ms)}",
                     fontsize=10)
        ax.set_xlabel("fill2 (matrica B, nuosekliai), ms")
        ax.set_ylabel("tankis")
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(sizes):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title("Nuoseklus matricos B pildymas procesoriuje: visų paleidimų skirstinys\n"
                       "(variantai seq, openmp, opencl, mpi-opt; mpi-base neįtrauktas: jo fill2 apima Bcast)"),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Plots for the report from bench CSVs.")
    ap.add_argument("--input", "-i", nargs="+", default=["results/csv"], metavar="PATH",
                    help="CSV directories and/or files (default: results/csv)")
    ap.add_argument("--output", "-o", default="results", metavar="DIR",
                    help="report directory; plots go to <DIR>/plots (default: results)")
    ap.add_argument("--sizes", type=int_set, metavar="LIST", help="e.g. 2000-4000 or 2000,4000,8000-16000")
    ap.add_argument("--workers", type=int_set, metavar="LIST", help="e.g. 1 or 1-8")
    ap.add_argument("--title", default="", help="stage title, first line of every plot title")
    ap.add_argument("--vline", type=vline, action="append", default=[], metavar="X[:LABEL]",
                    help="vertical marker on the size plots, may be repeated")
    ap.add_argument("--plots", nargs="+", choices=PLOTS, default=list(PLOTS), metavar="PLOT",
                    help="subset of: " + " ".join(PLOTS))
    ap.add_argument("--e-ratio", type=float, metavar="R",
                    help="E-core speed / P-core speed, adds <variant>-amdahl-hetero.png")
    ap.add_argument("--p-cores", type=int, default=8, metavar="P", help="performance cores (default: 8)")
    args = ap.parse_args()

    def title(text):
        return f"{args.title}\n{text}" if args.title else text

    paths = []
    for arg in args.input:
        p = Path(arg)
        if p.is_dir():
            paths += sorted(p.glob("*.csv"))
        elif p.is_file():
            paths.append(p)
        else:
            sys.exit(f"Input not found: {arg}")
    if not paths:
        sys.exit(f"No CSV files found in: {' '.join(args.input)}")
    data, fill2 = load(paths, args.sizes, args.workers)
    if not data:
        sys.exit("No rows left after the --sizes / --workers filters")
    out_dir = Path(args.output) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)  # existing files are kept, same names overwritten
    prefix = f"{out_dir}/"
    print(f"Using {len(paths)} CSV file(s) from: " + ", ".join(sorted({str(p.parent) for p in paths})))
    names = sorted({k[0] for k in data})
    sizes = sorted({k[1] for k in data})
    scaling = [n for n in names if n.startswith(("openmp-", "mpi-"))]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{prefix}{name}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 1) time vs workers, one figure per variant, one line per size
    if "time" in args.plots:
        for name in scaling:
            for yscale, suffix in SCALES:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for s in sizes:
                    pts = sorted((k[2], v["total"]) for k, v in data.items()
                                 if k[0] == name and k[1] == s)
                    if pts:
                        ax.plot(*zip(*pts), label=f"{s}x{s}")
                ax.set(title=title(f"{name}: vykdymo laikas pagal gijų (procesų) skaičių"),
                       xlabel="gijos / procesai", ylabel="laikas, s", yscale=yscale)
                ax.grid(True, alpha=0.3)
                legend_outside(ax, title="matricos dydis")
                save(fig, f"{name}-time{suffix}.png")

    # 2) speedup vs workers, one figure per variant, one line per size:
    #    S(w) = T(variant, n, 1 worker) / T(variant, n, w workers)
    #    Each variant is compared with itself on 1 thread / process (not with seq-base),
    #    so the plot shows parallel scaling only and is comparable with the "ideal" line.
    #    Sizes without a 1-worker run have no baseline and are skipped.
    if "speedup" in args.plots:
        for name in scaling:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            max_w = 1
            for s in sizes:
                t1 = data.get((name, s, 1))
                if t1 is None:
                    continue
                pts = sorted((k[2], t1["total"] / v["total"]) for k, v in data.items()
                             if k[0] == name and k[1] == s)
                if len(pts) > 1:
                    ax.plot(*zip(*pts), label=f"{s}x{s}")
                    max_w = max(max_w, pts[-1][0])
            if max_w == 1:
                plt.close(fig)
                continue
            ax.plot([1, max_w], [1, max_w], color="black", linestyle="--", marker="", alpha=0.4,
                    label="idealus")
            ax.set(title=title(f"{name}: pagreitėjimas, palyginti su 1 gija (procesu)"),
                   xlabel="gijos / procesai", ylabel="pagreitėjimas")
            ax.grid(True, alpha=0.3)
            legend_outside(ax, title="matricos dydis")
            save(fig, f"{name}-speedup.png")

    # 3) time vs size for every variant (best worker count)
    def best(name, s):
        vs = [v for k, v in data.items() if k[0] == name and k[1] == s]
        return min(vs, key=lambda v: v["total"]) if vs else None

    def size_points(name):
        return [(s, b["total"]) for s in sizes if (b := best(name, s))]

    ws = {k[2] for k in data}
    if len(ws) > 1:
        workers_note = "geriausias gijų (procesų) skaičius"
    elif all(n.startswith("opencl-") for n in names):
        workers_note = f"GPU skaičiavimo vienetų: {min(ws)}"
    else:
        workers_note = f"gijų (procesų): {min(ws)}"

    def vlines(ax, to_x):
        for x, label in args.vline:
            ax.axvline(to_x(x), color="#c0392b", linestyle="--", marker="", linewidth=1.2,
                       label=label or f"n = {dec(x)}")

    if "size" in args.plots:
        # 3a) log-log: T ∝ n^a is a straight line with slope a. Reference ∝ n³ through
        #     the geometric mean of the times at the smallest size.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in names:
            pts = size_points(name)
            if pts:
                ax.plot(*zip(*pts), label=name)
        s0 = sizes[0]
        t0 = [t for name in names for s, t in size_points(name) if s == s0]
        if len(sizes) > 1 and t0:
            y0 = math.exp(statistics.mean(math.log(t) for t in t0))
            ax.plot(sizes, [y0 * (s / s0) ** 3 for s in sizes], color="black", linestyle=":",
                    marker="", linewidth=1.2, label="∝ n³ (nuolydis 3)")
        vlines(ax, lambda x: x)
        ax.set(title=title(f"Vykdymo laikas pagal matricos dydį, {workers_note}"),
               xlabel="n = k = p (logaritminė skalė)", ylabel="laikas, s (logaritminė skalė)",
               xscale="log", yscale="log")
        ticks = spread_log(sizes)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        plain_ticks(ax.yaxis)
        ax.grid(True, which="both", alpha=0.3)
        legend_outside(ax)
        save(fig, "size.png")

        # 3b) linear axes, X = number of multiply-adds n·k·p (= n³): time is proportional
        #     to the work for a kernel whose speed does not depend on n
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name in names:
            pts = size_points(name)
            if pts:
                ax.plot([s ** 3 for s, _ in pts], [t for _, t in pts], label=name)
        vlines(ax, lambda x: x ** 3)
        ops = [s ** 3 for s in sizes]
        ticks = spread_ticks(ops, 0.09 * max(ops))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{sci(o)}\n({round(o ** (1 / 3))}³)" for o in ticks], fontsize=8)
        ax.set(title=title(f"Vykdymo laikas pagal operacijų skaičių, {workers_note}"),
               xlabel="daugybos ir sudėties operacijų skaičius n·k·p", ylabel="laikas, s")
        ax.grid(True, alpha=0.3)
        legend_outside(ax)
        save(fig, "size-linear.png")

    # 4) Amdahl's law. f = fill2 / total of the 1-worker run: by the task, matrix B is
    #    filled sequentially, the other phases are parallel. So f is measured, not fitted.
    #    S_amdahl(N) = 1 / (f + (1 - f) / N)
    def sizes_with_base(name):
        """sizes that have a 1-worker run and at least one run with more workers"""
        return [s for s in sizes if (name, s, 1) in data
                and any(k[0] == name and k[1] == s and k[2] > 1 for k in data)]

    def serial_fraction(name, s):
        t1 = data[(name, s, 1)]
        return t1["fill2"] / t1["total"]

    def measured_speedup(name, s):
        t1 = data[(name, s, 1)]["total"]
        return sorted((k[2], t1 / v["total"]) for k, v in data.items() if k[0] == name and k[1] == s)

    def size_colors(ss):
        cmap = plt.get_cmap("viridis")
        return {s: cmap(0.9 * i / max(1, len(ss) - 1)) for i, s in enumerate(ss)}

    def n_eff(n_workers):
        """effective number of P-cores: first P workers on P-cores, the rest on E-cores"""
        return min(n_workers, args.p_cores) + args.e_ratio * max(n_workers - args.p_cores, 0)

    def amdahl_plot(name, ss, hetero):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        cols = size_colors(ss)
        max_w = max(k[2] for k in data if k[0] == name)
        grid = [1 + (max_w - 1) * i / 200 for i in range(201)]
        cores = n_eff if hetero else (lambda w: w)
        for s in ss:
            f = serial_fraction(name, s)
            ax.plot(*zip(*measured_speedup(name, s)), color=cols[s], linestyle="-", marker="o",
                    markersize=4, linewidth=1.2, label=f"{s}x{s}, f = {dec(100 * f, '.2f')} %")
            ax.plot(grid, [1 / (f + (1 - f) / cores(w)) for w in grid], color=cols[s],
                    linestyle="--", marker="", linewidth=1, alpha=0.8)
        ax.plot(grid, [cores(w) for w in grid], color="black", linestyle=":", marker="",
                label="idealus" + (" (N_eff)" if hetero else ""))
        ax.plot([], [], color="gray", linestyle="-", marker="o", markersize=4, label="išmatuota")
        ax.plot([], [], color="gray", linestyle="--", marker="", label="Amdalo dėsnis")
        model = (f"Amdalo dėsnis, N_eff = min(N, {args.p_cores}) + {dec(args.e_ratio)}·max(N − {args.p_cores}, 0)"
                 if hetero else "Amdalo dėsnis, f = fill2 / total (1 gija)")
        ax.set(title=title(f"{name}: išmatuotas pagreitėjimas ir {model}"), xlabel="gijos / procesai",
               ylabel="pagreitėjimas")
        ax.title.set_fontsize(10)
        ax.grid(True, alpha=0.3)
        legend_outside(ax, fontsize=8)
        save(fig, f"{name}-amdahl{'-hetero' if hetero else ''}.png")

    if "amdahl" in args.plots:
        for name in scaling:
            ss = sizes_with_base(name)
            if ss:
                amdahl_plot(name, ss, hetero=False)
                if args.e_ratio is not None:
                    amdahl_plot(name, ss, hetero=True)

    # 5) phases: stacked time of fill1 (parallel), fill2 (sequential), mul (parallel)
    if "phases" in args.plots:
        phase_style = (("fill1", "matricos A pildymas (lygiagretus)", "#9fb4cc"),
                       ("fill2", "matricos B pildymas (nuoseklus)", "#c0392b"),
                       ("mul", "daugyba (lygiagreti)", "#2e5e8c"))
        for name in scaling:
            ss = [s for s in sizes if sum(1 for k in data if k[0] == name and k[1] == s) > 1]
            if not ss:
                continue
            cols = min(4, len(ss))
            rows = math.ceil(len(ss) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
            for ax, s in zip(axes.flat, ss):
                ws = sorted(k[2] for k in data if k[0] == name and k[1] == s)
                xs = [str(w) for w in ws]
                bottom = [0.0] * len(ws)
                for col, label, color in phase_style:
                    vals = [data[(name, s, w)][col] for w in ws]
                    ax.bar(xs, vals, bottom=bottom, color=color, label=label)
                    bottom = [b + v for b, v in zip(bottom, vals)]
                ax.set_title(f"n = {s}", fontsize=10)
                ax.set_xlabel("gijos / procesai", fontsize=8)
                ax.set_ylabel("laikas, s", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, axis="y", alpha=0.3)
            for ax in list(axes.flat)[len(ss):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
                       bbox_to_anchor=(0.5, 1.0))
            fig.suptitle(title(f"{name}: fazių laikas pagal gijų (procesų) skaičių"), y=1.06, fontsize=12)
            save(fig, f"{name}-phases.png")

    # 6) serial fraction vs size and the Amdahl limit 1/f vs the best measured speedup
    if "fraction" in args.plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in scaling:
            ss = sizes_with_base(name)
            if not ss:
                continue
            fs = [serial_fraction(name, s) for s in ss]
            line, = ax1.plot(ss, [100 * f for f in fs], label=name)
            ax2.plot(ss, [1 / f for f in fs], color=line.get_color(), linestyle="--", marker="",
                     label=f"{name}: riba 1/f")
            ax2.plot(ss, [max(sp for _, sp in measured_speedup(name, s)) for s in ss],
                     color=line.get_color(), linestyle="-", marker=line.get_marker(),
                     label=f"{name}: geriausias išmatuotas")
        ax1.set(title="Nuoseklioji dalis f = fill2 / total (1 gija)", xlabel="n = k = p", ylabel="f, %")
        ax2.set(title="Amdalo riba S(∞) = 1/f ir geriausias išmatuotas pagreitėjimas", xlabel="n = k = p",
                ylabel="pagreitėjimas", yscale="log")
        plain_ticks(ax2.yaxis)
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        legend_outside(ax2, fontsize=8)
        fig.suptitle(title("Nuoseklioji dalis: didesnėms matricoms – mažesnė (Gustafsono dėsnis)"), fontsize=12)
        save(fig, "amdahl-fraction.png")

    # 7) Karp-Flatt metric: e(N) = (1/S - 1/N) / (1 - 1/N) is the serial fraction that would
    #    explain the measured speedup S on N workers. e == f: Amdahl explains everything;
    #    e > f or e growing with N: other losses (memory, synchronization, slower cores).
    if "karpflatt" in args.plots:
        for name in scaling:
            ss = sizes_with_base(name)
            if not ss:
                continue
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            cols = size_colors(ss)
            kf = {s: [(w, (1 / sp - 1 / w) / (1 - 1 / w)) for w, sp in measured_speedup(name, s) if w > 1]
                  for s in ss}
            for s in ss:
                ax1.plot(*zip(*[(w, 100 * e) for w, e in kf[s]]), color=cols[s], linestyle="-",
                         marker="o", markersize=4, label=f"{s}x{s}")
            ax1.set(title="e(N) kiekvienam matricos dydžiui", xlabel="gijos / procesai", ylabel="e, %")
            legend_outside(ax1, fontsize=8, title="matricos dydis")
            ws = sorted({w for s in ss for w, _ in kf[s]})
            for w in ws:
                pts = [(s, 100 * e) for s in ss for ww, e in kf[s] if ww == w]
                ax2.plot(*zip(*pts), label=f"e, N = {w}")
            ax2.plot(ss, [100 * serial_fraction(name, s) for s in ss], color="black", linestyle="--",
                     marker="", label="išmatuota f (fill2)")
            ax2.set(title="e ir išmatuota f", xlabel="n = k = p", ylabel="%")
            legend_outside(ax2, fontsize=8)
            for ax in (ax1, ax2):
                ax.grid(True, alpha=0.3)
            fig.suptitle(title(f"{name}: Karpo–Flato metrika (eksperimentiškai nustatyta nuoseklioji dalis)"),
                         fontsize=12)
            save(fig, f"{name}-karpflatt.png")

    # 8) distribution ("bell") of the sequential fill time of matrix B
    if "fill2" in args.plots:
        plot_fill2_distribution(fill2, f"{prefix}fill2-distribution.png", title)
        # median fill time vs number of elements k·p (= n²): linear, fit through the origin
        pts = [(n * n, statistics.median(v)) for n, v in sorted(fill2.items())]
        if len(pts) > 1:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            xs, ys = zip(*pts)
            c = sum(x * y for x, y in pts) / sum(x * x for x in xs)  # seconds per element
            ax.plot(xs, [1000 * y for y in ys], linestyle="", marker="o", label="mediana")
            ax.plot([0, max(xs)], [0, 1000 * c * max(xs)], color="#c0392b", linestyle="--", marker="",
                    label=f"tiesė per 0:\n{dec(c * 1e9, '.2f')} ns / elementui")
            ticks = spread_ticks(list(xs), 0.1 * max(xs))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{sci(x)}\n({round(x ** 0.5)}²)" for x in ticks], fontsize=8)
            ax.set(title=title("Matricos B pildymo laikas pagal elementų skaičių (mediana)"),
                   xlabel="matricos B elementų skaičius k·p", ylabel="fill2, ms")
            ax.grid(True, alpha=0.3)
            legend_outside(ax)
            save(fig, "fill2-elements.png")

    print(f"Plots saved: {out_dir}/")


if __name__ == "__main__":
    main()
