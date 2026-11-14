#!/usr/bin/env python3
"""Summary figures for the report conclusions (reports/ataskaita/src/4_isvados.md).

Usage: scripts/plot_final.py [output dir]      default: reports/ataskaita/img/final
  kelias.png  the path seq-base -> best variants at n = 4000: total time of every step
              on one log axis, with the factor gained by each step
  fazes.png   share of every phase (fill1, fill2, mul) in the total time at n = 4000:
              the sequential fill of B becomes a large share once the rest is fast
Medians from results/csv/*.csv (scripts/plot.py load()). Labels in Lithuanian.
"""
import locale
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "lt_LT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass

N = 4000
STEPS = [  # (label, variant, workers, colour)
    ("seq-base\n(1 gija, i-j-z)", "seq-base", 1, "#8a96a8"),
    ("seq-opt\n(1 gija, blokai)", "seq-opt", 1, "#e9a23b"),
    ("openmp-opt2\n(8 gijos)", "openmp-opt2", 8, "#2f6690"),
    ("mpi-opt\n(8 procesai)", "mpi-opt", 8, "#6a4c93"),
    ("opencl regblock\n(GPU)", "opencl-regblock", 19, "#3f8f3a"),
]


def dec(x, fmt):
    return format(x, fmt).replace(".", ",")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/ataskaita/img/final")
    out.mkdir(parents=True, exist_ok=True)
    data, _ = plot.load(sorted(Path("results/csv").glob("*.csv")))
    rows = [(label, data[(v, N, w)], color) for label, v, w, color in STEPS]

    # 1) the path
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = list(range(len(rows)))[::-1]
    base = rows[0][1]["total"]
    for yi, (label, d, color) in zip(y, rows):
        t = d["total"]
        ax.barh(yi, t, color=color, height=0.6)
        ax.text(t * 1.15, yi, f"{dec(t, '.3f' if t < 10 else '.2f')} s  ·  {dec(base / t, '.0f')}× greičiau už seq-base" if t < base
                else f"{dec(t, '.1f')} s", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: dec(x, "g")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("visas vykdymo laikas, s (logaritminė skalė)")
    ax.set_title(f"Kelias nuo seq-base iki geriausių variantų, n = {N}", fontsize=11, loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "kelias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) phase shares
    fig, ax = plt.subplots(figsize=(9, 3.6))
    phases = (("fill1", "A pildymas", "#9fb4cc"), ("fill2", "B pildymas (nuosekliai)", "#c0392b"),
              ("mul", "daugyba", "#2e5e8c"))
    for yi, (label, d, _) in zip(y[1:], rows[1:]):
        left = 0
        for key, name, color in phases:
            share = 100 * d[key] / d["total"]
            ax.barh(yi, share, left=left, color=color, height=0.6, label=name if yi == y[1] else None)
            if key == "fill2":  # white inside a wide red segment, dark after a narrow one
                inside = share > 8
                ax.text(left + share / 2 if inside else left + share + 1, yi, f"{dec(share, '.0f')} %", va="center",
                        ha="center" if inside else "left", fontsize=9, fontweight="bold",
                        color="white" if inside else "#172033")
            left += share
    ax.set_yticks(y[1:])
    ax.set_yticklabels([r[0] for r in rows[1:]], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("dalis viso laiko, %")
    ax.set_title(f"Fazių dalys, n = {N}: kuo greitesnė lygiagreti dalis, tuo svarbesnė nuosekli", fontsize=11, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fazes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
