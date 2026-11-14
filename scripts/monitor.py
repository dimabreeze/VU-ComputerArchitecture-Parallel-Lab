#!/usr/bin/env python3
"""CPU / GPU load during program runs: monitor screenshots and per-core telemetry.

For every scenario (see SCENARIOS):
  1. `macmon pipe` records, every 100 ms, the load and frequency of every CPU core
     (E and P clusters), GPU load / frequency, power, memory pressure and swap.
     macmon needs no sudo.
  2. `btop` runs in a detached tmux session (started a few seconds before the program,
     so its history graph is filled); in the middle of the multiplication its screen is
     captured with colours (tmux capture-pane -e) and rendered to PNG by `freeze`.
     The process list is filtered to commands that start with "build/", so only the
     measured program is shown. For GPU scenarios the `macmon` screen is captured too.
  3. the program's own report gives the phase durations (fill1, fill2, mul), used to
     mark the phases on the heat map (scripts/plot_monitor.py).

Output: results/monitor/<scenario>.jsonl (macmon samples), <scenario>.json (scenario,
start time, phases), docs/img/screens/<scenario>-btop.png [-macmon.png].
Everything is written to a temporary directory first and moved into the project at
the end (the project lives in OneDrive, which loses lines of files kept open while
writing).

Usage: scripts/monitor.py [scenario ...]     (default: all; run on an idle machine)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# name: (command, seconds after start to capture the screen, also capture macmon screen)
SCENARIOS = {
    "seq-base":          (["build/seq-base", "-s", "3000"], 16, False),
    "seq-opt":           (["build/seq-opt", "-s", "8000"], 8, False),
    "openmp-base-t8":    (["build/openmp-base", "-s", "4000", "-t", "8"], 18, False),
    "openmp-opt2-t1":    (["build/openmp-opt2", "-s", "8000", "-t", "1"], 9, False),
    "openmp-opt2-t8":    (["build/openmp-opt2", "-s", "16000", "-t", "8"], 13, False),
    "openmp-opt2-t12":   (["build/openmp-opt2", "-s", "16000", "-t", "12"], 12, False),
    "mpi-opt-np8":       (["mpirun", "-np", "8", "build/mpi-opt", "-s", "14000"], 12, False),
    "opencl-regblock":   (["build/opencl-opt", "-s", "16000", "--kernel", "src/opencl/opt/regblock.cl"], 7, True),
}
WARMUP = 8          # seconds of btop / macmon before the program starts
COOLDOWN = 2        # seconds after the program ends
TERM = (150, 30)    # tmux window size for the monitors

BTOP_OVERRIDES = {  # differences from `btop --default-config`
    "graph_symbol": '"block"',        # freeze's font has no braille glyphs
    "shown_boxes": '"cpu proc"',      # no network box (local IP), no disks
    "proc_per_core": "true",          # 800 % for 8 busy threads
    "show_battery": "false",
    "show_uptime": "false",
}


def sh(*args, **kw):
    return subprocess.run(args, check=True, **kw)


def btop_config(path):
    text = subprocess.run(["btop", "--default-config"], capture_output=True, text=True, check=True).stdout
    for key, value in BTOP_OVERRIDES.items():
        text = re.sub(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.M)
    path.write_text(text)


def capture(session, out_png):
    ans = out_png.with_suffix(".ans")
    with open(ans, "w") as f:
        subprocess.run(["tmux", "capture-pane", "-t", session, "-e", "-p"], stdout=f, check=True)
    for attempt in (1, 2):  # freeze once hung for minutes: time out and retry
        try:
            sh("freeze", str(ans), "--language", "ansi", "-o", str(out_png), "--window", "--font.size", "12",
               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=60)  # stdin open -> freeze waits for piped input
            break
        except subprocess.TimeoutExpired:
            print(f"   freeze timed out ({out_png.name}), attempt {attempt}")
    ans.unlink()


def hide_macmon_procs(session):
    """press "p" in macmon until its process list (other apps) disappears from the screen"""
    for _ in range(20):
        time.sleep(0.5)
        screen = subprocess.run(["tmux", "capture-pane", "-t", session, "-p"], capture_output=True, text=True).stdout
        if "PID" not in screen and "macmon" in screen:
            return
        if "PID" in screen:
            sh("tmux", "send-keys", "-t", session, "p")
    print("   warning: could not hide the macmon process list")


def phases(output):
    """phase durations from the program's report"""
    out = {}
    for key, pat in (("fill1", r"fill matrix1.*:\s*([\d.]+) s"), ("fill2", r"fill matrix2.*:\s*([\d.]+) s"),
                     ("mul", r"multiply.*:\s*([\d.]+) s"), ("total", r"total\s*:\s*([\d.]+) s")):
        m = re.search(pat, output)
        if m:
            out[key] = float(m.group(1))
    return out


def run(name, tmp):
    cmd, capture_at, with_macmon = SCENARIOS[name]
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    conf = tmp / "btop.conf"
    if not conf.exists():
        btop_config(conf)
    sessions = [f"btop-{name}"] + ([f"macmon-{name}"] if with_macmon else [])
    log = tmp / f"{name}.jsonl"
    with open(log, "w") as f:
        mon = subprocess.Popen(["macmon", "pipe", "-i", "100"], stdout=f, stderr=subprocess.DEVNULL)
    try:
        # the filter is a regex ("!" prefix) matched against the whole command: build/... only
        # (not btop, tmux or mpirun). -u 125: the history graph fills within the run
        sh("tmux", "new-session", "-d", "-s", sessions[0], "-x", str(TERM[0]), "-y", str(TERM[1]),
           f"btop -c {conf} -u 125 -f '!build/.*'")
        if with_macmon:
            sh("tmux", "new-session", "-d", "-s", sessions[1], "-x", str(TERM[0]), "-y", str(TERM[1]), "macmon -i 250")
            hide_macmon_procs(sessions[1])
        time.sleep(WARMUP)
        t0 = time.time()
        # /usr/bin/time -p: CPU time (user + sys) of the program and all its children
        # (for mpirun: all ranks), so the average number of busy cores = cpu / real
        prog = subprocess.Popen(["/usr/bin/time", "-p"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(capture_at)
        if prog.poll() is not None:
            print(f"   warning: finished before the capture ({time.time() - t0:.1f} s)")
        capture(sessions[0], tmp / f"{name}-btop.png")
        if with_macmon:
            capture(sessions[1], tmp / f"{name}-macmon.png")
        output = prog.communicate()[0]
        t1 = time.time()
        time.sleep(COOLDOWN)
    finally:
        mon.terminate()
        mon.wait()
        for s in sessions:
            subprocess.run(["tmux", "kill-session", "-t", s], stderr=subprocess.DEVNULL)
    rusage = {k: float(v) for k, v in re.findall(r"^(real|user|sys)\s+([\d.]+)$", output, re.M)}
    meta = {"scenario": name, "command": cmd, "start": t0, "end": t1, "capture_at": capture_at,
            "phases": phases(output), "time": rusage, "returncode": prog.returncode}
    (tmp / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"   {meta['phases']} {rusage}", flush=True)


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"unknown scenario(s): {unknown}; known: {list(SCENARIOS)}")
    for tool in ("macmon", "btop", "tmux", "freeze"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found (brew install btop macmon tmux charmbracelet/tap/freeze)")
    tmp = Path(tempfile.mkdtemp(prefix="monitor-"))
    for n in names:
        run(n, tmp)
    data, img = ROOT / "results/monitor", ROOT / "docs/img/screens"
    data.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        if f.suffix in (".jsonl", ".json"):
            shutil.move(str(f), data / f.name)
        elif f.suffix == ".png":
            shutil.move(str(f), img / f.name)
    print(f"Saved: {data.relative_to(ROOT)}/, {img.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
