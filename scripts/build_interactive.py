#!/usr/bin/env python3
"""Builds the interactive version of the report: one self-contained HTML page.

Usage: scripts/build_interactive.py [--fragment PATH]
Input:  results/csv/*.csv        benchmark runs (medians and repeat counts per configuration)
        results/monitor/*.json(l) CPU / GPU load telemetry (scripts/monitor.py)
        results/memory/*.csv     memory latency, TLB and bandwidth (scripts/memory.sh)
        scripts/interactive/page.html   the page (HTML/CSS/JS) with a /*__DATA__*/ placeholder
Output: reports/interactive/index.html   standalone page: open it in a browser, no server or
                                          network needed (only the web fonts need a network)
        --fragment PATH                   the same page without <html>/<head>/<body> (the form
                                          the claude.ai Artifact publisher expects)
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "scripts/interactive/page.html"
OUT = ROOT / "reports/interactive/index.html"
AUTHOR = "Dmitrij Kosmakov"
REPO = "https://github.com/dimabreeze/VU-ComputerArchitecture-Parallel-Lab"


def r(x, d=4):
    return round(x, d)


def benchmarks():
    """variant -> n -> workers -> [fill1, fill2, mul, total, repeats] (medians)"""
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted((ROOT / "results/csv").glob("*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    key = (f"{row['tech']}-{row['variant']}", int(row["n"]), int(row["workers"]))
                    vals = [float(row[c]) for c in ("fill1", "fill2", "mul", "total")]
                except (KeyError, TypeError, ValueError):
                    continue
                runs[key]["v"].append(vals)
    out = defaultdict(lambda: defaultdict(dict))
    for (name, n, w), d in runs.items():
        cols = list(zip(*d["v"]))
        out[name][n][w] = [r(statistics.median(c), 6) for c in cols] + [len(d["v"])]
    return {k: {n: dict(sorted(ws.items())) for n, ws in sorted(v.items())} for k, v in sorted(out.items())}


def monitor():
    """scenario -> time series of per-core load (%), frequencies, memory, phases"""
    out = {}
    for meta_path in sorted((ROOT / "results/monitor").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        t0 = meta["start"]
        s = {k: [] for k in ("t", "e", "p", "g", "fp", "fe", "fg", "ram", "swap")}
        with open(meta_path.with_suffix(".jsonl")) as f:
            for line in f:
                if not line.startswith("{"):
                    continue
                d = json.loads(line)
                tiers = {t["label"]: t for t in d["cpu_tiers"]}
                s["t"].append(r(datetime.fromisoformat(d["timestamp"]).timestamp() - t0, 2))
                s["e"].append([round(100 * c["active_ratio"]) for c in tiers["E"]["cores"]])
                s["p"].append([round(100 * c["active_ratio"]) for c in tiers["P"]["cores"]])
                s["g"].append(round(100 * d["gpu_active_ratio"]))
                s["fp"].append(round(d["pcpu_freq_mhz"]))
                s["fe"].append(round(d["ecpu_freq_mhz"]))
                s["fg"].append(round(d["gpu_freq_mhz"]))
                s["ram"].append(r(d["memory"]["ram_usage"] / 2**30, 2))
                s["swap"].append(r(d["memory"]["swap_usage"] / 2**30, 2))
        tm = meta.get("time", {})
        ph = meta["phases"]
        s["cmd"] = " ".join(meta["command"])
        s["phases"] = ph
        s["end"] = r(meta["end"] - meta["start"], 2)
        s["fill1_start"] = r(s["end"] - ph.get("total", 0), 2)  # phases aligned to the end of the run
        s["cores"] = r((tm.get("user", 0) + tm.get("sys", 0)) / tm["real"], 2) if tm.get("real") else None
        out[meta_path.stem] = s
    return out


def memory():
    def read(name, key):
        pts = defaultdict(list)
        path = ROOT / f"results/memory/{name}.csv"
        if not path.exists():
            return []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["mode"] == "mode":
                    continue
                pts[int(row[key])].append(float(row["value"]))
        return [[k, r(statistics.median(v) if name != "bw" else max(v), 3)] for k, v in sorted(pts.items())]
    return {"lat": read("lat", "bytes"), "tlb": read("tlb", "pages"), "bw": read("bw", "threads")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="also write the page without html/head/body")
    args = ap.parse_args()
    data = {"author": AUTHOR, "repo": REPO, "generated": datetime.now().strftime("%Y-%m-%d"),
            "bench": benchmarks(), "mon": monitor(), "mem": memory()}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    page = PAGE.read_text().replace("/*__DATA__*/", f"const DATA = {blob};")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("<!doctype html>\n<html lang=\"lt\">\n<head>\n<meta charset=\"utf-8\">\n"
                   "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                   "<style>html,body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
                   "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    print(f"{OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.2f} MB, data {len(blob) / 1e6:.2f} MB)")
    if args.fragment:
        args.fragment.write_text(page)
        print(args.fragment)


if __name__ == "__main__":
    sys.exit(main())
