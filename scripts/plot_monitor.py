#!/usr/bin/env python3
"""Heat maps of CPU / GPU load from scripts/monitor.py runs.

Usage: scripts/plot_monitor.py [results dir] [output dir]
  defaults: results/monitor  docs/img/load
For every <scenario>.jsonl (macmon samples) + <scenario>.json (start time, phases):
  <scenario>.png with three panels sharing the time axis (0 = program start):
    1. load of every core (E0-E3, P0-P7) and of the GPU, 0-100 %, as a heat map,
       with the program phases (fill1 parallel, fill2 sequential, mul parallel) on top;
    2. clock frequency of the P cluster, the E cluster and the GPU;
    3. used RAM and swap.
The phases come from the program's own report; they are aligned to the end of the run
(start of fill1 = end - total), because mpirun / OpenCL start-up happens before fill1.
Labels are in Lithuanian, numbers use the decimal comma.
"""
import json
import locale
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

try:
    locale.setlocale(locale.LC_NUMERIC, "lt_LT.UTF-8")
    plt.rcParams["axes.formatter.use_locale"] = True
except locale.Error:
    pass

TITLES = {
    "seq-base": "seq-base, n = 3000: viena gija",
    "seq-opt": "seq-opt, n = 8000: viena gija",
    "openmp-base-t8": "openmp-base, n = 4000, 8 gijos",
    "openmp-opt2-t1": "openmp-opt2, n = 8000, 1 gija",
    "openmp-opt2-t8": "openmp-opt2, n = 16000, 8 gijos",
    "openmp-opt2-t12": "openmp-opt2, n = 16000, 12 gijų",
    "mpi-opt-np8": "mpi-opt, n = 14000, 8 procesai",
    "opencl-regblock": "opencl-opt (regblock), n = 16000, GPU",
}
PHASES = (("fill1", "A pildymas", "#9fb4cc"),
          ("fill2", "B pildymas (visada nuosekliai, CPU)", "#c0392b"),
          ("mul", "daugyba", "#2e5e8c"))


def load(jsonl, t0):
    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            t = datetime.fromisoformat(r["timestamp"]).timestamp() - t0
            tiers = {tier["label"]: tier for tier in r["cpu_tiers"]}
            rows.append(dict(
                t=t,
                e=[c["active_ratio"] for c in tiers["E"]["cores"]],
                p=[c["active_ratio"] for c in tiers["P"]["cores"]],
                gpu=r["gpu_active_ratio"],
                fp=r["pcpu_freq_mhz"], fe=r["ecpu_freq_mhz"], fg=r["gpu_freq_mhz"],
                ram=r["memory"]["ram_usage"] / 2**30, swap=r["memory"]["swap_usage"] / 2**30))
    return rows


def plot(name, src, out):
    meta = json.loads((src / f"{name}.json").read_text())
    rows = load(src / f"{name}.jsonl", meta["start"])
    if len(rows) < 3:
        print(f"{name}: too few samples, skipped")
        return
    ts = [r["t"] for r in rows]
    step = (ts[-1] - ts[0]) / (len(ts) - 1)
    ne, np_ = len(rows[0]["e"]), len(rows[0]["p"])
    # rows of the heat map, top to bottom: GPU, P7..P0, E3..E0
    labels = ["GPU"] + [f"P{i}" for i in reversed(range(np_))] + [f"E{i}" for i in reversed(range(ne))]
    grid = [[r["gpu"] for r in rows]] + [[r["p"][i] for r in rows] for i in reversed(range(np_))] + \
           [[r["e"][i] for r in rows] for i in reversed(range(ne))]

    fig, (ax, axf, axm) = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                                       gridspec_kw={"height_ratios": [3.2, 1.1, 1.0]})
    im = ax.imshow([[100 * v for v in row] for row in grid], aspect="auto", cmap="inferno", vmin=0, vmax=100,
                   extent=(ts[0] - step / 2, ts[-1] + step / 2, len(labels) - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for y in (0.5, 0.5 + np_):  # separators GPU | P | E
        ax.axhline(y, color="white", linewidth=1.2)
    ax.set_ylabel("branduolys")
    cb = fig.colorbar(im, ax=[ax, axf, axm], location="right", fraction=0.025, pad=0.01)
    cb.set_label("apkrova, %")

    # phases, aligned to the end of the run
    ph = meta["phases"]
    t = (meta["end"] - meta["start"]) - ph.get("total", sum(ph.get(k, 0) for k, _, _ in PHASES))
    span = {}
    for key, label, color in PHASES:
        d = ph.get(key, 0)
        span[key] = (t, t + d)
        ax.add_patch(Rectangle((t, -1.45), d, 0.8, color=color, clip_on=False))
        if d > 0.15 * (ts[-1] - ts[0]):
            ax.text(t + d / 2, -1.05, label, ha="center", va="center", fontsize=8, color="white", clip_on=False)
        t += d
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in PHASES], loc="lower center",
               bbox_to_anchor=(0.45, -0.03), ncol=3, fontsize=8, frameon=False)
    # average number of busy cores of the program itself: CPU time (user + sys, from
    # /usr/bin/time, children included) / wall time
    tm = meta.get("time", {})
    note = ""
    if tm.get("real"):
        cores = (tm.get("user", 0) + tm.get("sys", 0)) / tm["real"]
        note = f"\nprogramos procesoriaus laikas / trukmė = {cores:.1f} branduolio (vidutiniškai)".replace(".", ",")
    ax.set_title(TITLES.get(name, name) + note, fontsize=11, loc="left", pad=22)

    axf.plot(ts, [r["fp"] for r in rows], label="P branduoliai", color="#e9a23b")
    axf.plot(ts, [r["fe"] for r in rows], label="E branduoliai", color="#5b8e7d")
    axf.plot(ts, [r["fg"] for r in rows], label="GPU", color="#9b2c5a")
    axf.set_ylabel("dažnis, MHz")
    axf.legend(loc="upper left", fontsize=7, ncol=3)
    axf.grid(True, alpha=0.3)

    axm.plot(ts, [r["ram"] for r in rows], label="naudojama RAM", color="#2e5e8c")
    axm.plot(ts, [r["swap"] for r in rows], label="swap", color="#c0392b")
    axm.set_ylabel("GB")
    axm.set_ylim(0, 17)
    axm.legend(loc="upper left", fontsize=7, ncol=2)
    axm.grid(True, alpha=0.3)
    axm.set_xlabel("laikas nuo programos paleidimo, s")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{out / name}.png")


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/monitor")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img/load")
    for meta in sorted(src.glob("*.json")):
        plot(meta.stem, src, out)


if __name__ == "__main__":
    main()
