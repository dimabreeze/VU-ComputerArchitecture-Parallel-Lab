#!/usr/bin/env python3
"""Plots of the memory microbenchmarks (build/tools/memlat, docs/memory.md).

Usage: scripts/plot_memory.py [results dir] [output dir]
  defaults: results/memory  docs/img/memory
Input:  lat.csv, tlb.csv, bw.csv (header mode,bytes,pages,threads,value); a file may hold
        several runs (repeated headers are skipped), points are reduced to the median.
Output: lat.png (latency vs working set), tlb.png (latency vs pages), bw.png (bandwidth).
Labels are in Lithuanian, numbers use the decimal comma.
"""
import csv
import locale
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "lt_LT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass

# machine parameters (sysctl, docs/machine.md)
L1D = 128 << 10
L2 = 16 << 20
TLB_ENTRIES = 3072
PAGE = 16 << 10


def dec(x, fmt="g"):
    return format(x, fmt).replace(".", ",")


def size_label(b):
    return f"{dec(b / (1 << 20))} MB" if b >= 1 << 20 else f"{dec(b / 1024)} KB"


def read(path, reduce=statistics.median):
    """(bytes or pages or threads) -> value reduced over runs (median by default)"""
    pts = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["mode"] == "mode":  # header of a repeated run
                continue
            key = {"lat": "bytes", "tlb": "pages", "bw": "threads"}[row["mode"]]
            pts[int(row[key])].append(float(row["value"]))
    return {k: reduce(v) for k, v in sorted(pts.items())}


def finish(fig, ax, path):
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/memory")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img/memory")
    out.mkdir(parents=True, exist_ok=True)

    # 1) latency vs working set
    lat = read(src / "lat.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(list(lat), list(lat.values()), marker="o", color="#2e5e8c", label="išmatuota")
    for x, label, color in ((L1D, "L1d = 128 KB", "#5b8e7d"), (L2, "L2 = 16 MB", "#e9a23b"),
                            (TLB_ENTRIES * PAGE, "TLB aprėptis ≈ 48 MB", "#c0392b")):
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, label=label)
    ax.set(xscale="log", yscale="log", xlabel="darbinio duomenų rinkinio dydis (logaritminė skalė)",
           ylabel="vieno skaitymo laikas, ns (log.)",
           title="Atsitiktinio skaitymo delsa pagal duomenų kiekį (128 B eilutės, 1 gija)")
    ticks = [8 << 10, 64 << 10, 512 << 10, 4 << 20, 32 << 20, 256 << 20, 1 << 30]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: size_label(x)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: dec(y)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for x0, x1, name in ((8 << 10, L1D, "L1"), (256 << 10, 8 << 20, "L2"), (64 << 20, 1 << 30, "RAM")):
        xs = [b for b in lat if x0 <= b <= x1]
        if xs:
            y = statistics.median(lat[b] for b in xs)
            ax.annotate(f"{name}: {dec(y, '.2f' if y < 10 else '.0f')} ns", (xs[len(xs) // 2], y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color="#172033")
    ax.legend(loc="upper left", fontsize=8)
    finish(fig, ax, out / "lat.png")

    # 2) TLB: one line per page
    tlb = read(src / "tlb.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(list(tlb), list(tlb.values()), marker="o", color="#2e5e8c", label="išmatuota (3 paleidimų mediana)")
    ax.axvline(TLB_ENTRIES, color="#c0392b", linestyle="--", linewidth=1.2, label="3072 puslapiai = 48 MB")
    ax.axvspan(6144, max(tlb), color="#8a96a8", alpha=0.15, label="nestabili sritis")
    ax.set(xscale="log", xlabel="paliečiamų 16 KB puslapių skaičius (logaritminė skalė)",
           ylabel="vieno skaitymo laikas, ns",
           title="TLB: po vieną podėlio eilutę kiekviename puslapyje (1 gija)")
    ticks = [16, 64, 256, 1024, 3072, 8192, 32768]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper left", fontsize=8)
    finish(fig, ax, out / "tlb.png")

    # 3) sequential read bandwidth vs threads
    bw = read(src / "bw.csv", max)  # achievable bandwidth: best of the runs
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([str(t) for t in bw], list(bw.values()), color="#2e5e8c")
    for i, v in enumerate(bw.values()):
        ax.annotate(dec(v, ".0f"), (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.axhline(200, color="#c0392b", linestyle="--", linewidth=1.2, label="Apple skelbiama: 200 GB/s")
    ax.set(xlabel="gijų skaičius", ylabel="GB/s", ylim=(0, 220),
           title="Nuoseklaus skaitymo pralaidumas (1 GB masyvas)")
    ax.legend(loc="upper left", fontsize=8)
    finish(fig, ax, out / "bw.png")
    print(f"Plots saved: {out}/")


if __name__ == "__main__":
    main()
