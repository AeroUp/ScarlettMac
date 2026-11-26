#!/usr/bin/env python3
"""ScarlettMac HUD: what the current run is doing, at a glance.

Reads the newest runs/*.log the launcher writes, plus the Mac's own counters,
and draws a small text dashboard: run and loading progress bars, fps, frame
time and interval, draws, copy cost, the queue, and CPU / RAM / swap / GPU.

  tools/scarlett-hud.py           live, redraws every second in this terminal
  tools/scarlett-hud.py --once    one frame to stdout (the floating window uses this)
  tools/scarlett-hud.sh           opens it in an always-on-top window
"""
import glob, os, re, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
LOCK = os.path.join(ROOT, ".build", "run.lock")
WIDTH = 64
# Launch to first world frame has been 200-270 s on an M-series Mac; the
# loading bar is an estimate against this, and turns full once the world draws.
TYPICAL_WORLD_S = 230
WORLD_DRAWS = 500          # more draws a frame than the loading screens ever use

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HOST = re.compile(r"^(\S+)\s+INFO victory_gpu::host: ([\d.]+) fps, ([\d.]+) ms/frame "
                  r"\(([\d.]+) preparing: copies ([\d.]+) ms ([\d.]+) MB, textures ([\d.]+) ms ([\d.]+) MB; "
                  r"([\d.]+) on the GPU.*?\), (\d+) draws/frame(?:, dropped (\d+))?")
INTERVAL = re.compile(r"frame interval interval=([\d.]+)ms")
FRAME_NO = re.compile(r"frame=(\d+)")
CRASH = re.compile(r"unhandled exception (0x[0-9a-f]+) at (0x[0-9a-f]+)")
SPARK = " ▁▂▃▄▅▆▇█"


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return ""


def bar(frac, width=30):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + ("▌" if frac * width - full >= 0.5 else "") + "·" * (width - full - (1 if frac * width - full >= 0.5 else 0))


def spark(values, width=40):
    values = values[-width:]
    if not values:
        return ""
    hi = max(values) or 1
    return "".join(SPARK[min(8, int(v / hi * 8 + 0.5))] for v in values)


def ts(s):
    try:
        return datetime.strptime(s[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def tail(path, nbytes=4_000_000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "replace")


def launchers():
    """Every world-run/scarlett-run launcher: (tag, planned seconds, pid)."""
    out = []
    for line in sh(["ps", "-axo", "pid=,command="]).splitlines():
        if "scarlett-run.sh" not in line or "zsh" not in line:
            continue
        tag = re.search(r"--tag (\S+)", line)
        sec = re.search(r"--seconds (\d+)", line)
        out.append((tag.group(1) if tag else "?", int(sec.group(1)) if sec else None, int(line.split()[0])))
    return out


def lock_owner():
    try:
        return int(open(os.path.join(LOCK, "pid")).read().strip())
    except Exception:
        return None


def run_state():
    logs = sorted(glob.glob(os.path.join(RUNS, "*.log")), key=os.path.getmtime)
    if not logs:
        return None
    path = logs[-1]
    name = os.path.basename(path)[:-4]
    m = re.match(r"(\d{8}-\d{6})-(.*)", name)
    started = time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S")) if m else os.path.getmtime(path)
    tag = m.group(2) if m else name
    text = ANSI.sub("", tail(path))
    samples, intervals, frame, crash, first_world = [], [], None, None, None
    for line in text.splitlines():
        h = HOST.match(line)
        if h:
            s = dict(t=ts(h.group(1)), fps=float(h.group(2)), ms=float(h.group(3)), prep=float(h.group(4)),
                     copy_ms=float(h.group(5)), copy_mb=float(h.group(6)), tex_ms=float(h.group(7)),
                     tex_mb=float(h.group(8)), gpu=float(h.group(9)), draws=int(h.group(10)),
                     dropped=int(h.group(11) or 0))
            samples.append(s)
            if first_world is None and s["draws"] >= WORLD_DRAWS and s["t"]:
                first_world = s["t"]
            continue
        i = INTERVAL.search(line)
        if i:
            intervals.append(float(i.group(1)))
        f = FRAME_NO.search(line)
        if f:
            frame = int(f.group(1))
        c = CRASH.search(line)
        if c:
            crash = f"{c.group(1)} at {c.group(2)}"
    planned = None
    live = False
    for t, sec, pid in launchers():
        if t == tag:
            planned = sec or planned
            live = live or pid == lock_owner()
    live = live or (time.time() - os.path.getmtime(path) < 20 and lock_owner() is not None)
    return dict(path=path, tag=tag, started=started, samples=samples, intervals=intervals, frame=frame,
                crash=crash, first_world=first_world, planned=planned, live=live, mtime=os.path.getmtime(path))


def system():
    ncpu = int(sh(["sysctl", "-n", "hw.ncpu"]) or 1)
    memsize = int(sh(["sysctl", "-n", "hw.memsize"]) or 1)
    page = 16384
    vm = {}
    for line in sh(["vm_stat"]).splitlines():
        if "page size of" in line:
            page = int(re.search(r"(\d+) bytes", line).group(1))
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            vm[k.strip()] = int(v)
    used = (vm.get("Pages active", 0) + vm.get("Pages wired down", 0) +
            vm.get("Pages occupied by compressor", 0)) * page
    swap = re.search(r"used = ([\d.]+)M", sh(["sysctl", "-n", "vm.swapusage"]))
    total_cpu, title_cpu, title_rss, wine_cpu = 0.0, 0.0, 0, 0.0
    for line in sh(["ps", "-axo", "%cpu=,rss=,command="]).splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        cpu, rss, cmd = float(parts[0]), int(parts[1]), parts[2]
        total_cpu += cpu
        if "victory.exe" in cmd and "--execute" in cmd:
            title_cpu += cpu
            title_rss += rss * 1024
        elif "wineserver" in cmd:
            wine_cpu += cpu
    gpu = re.search(r'"Device Utilization %"=(\d+)', sh(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]))
    gpu_mem = re.search(r'"In use system memory"=(\d+)', sh(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]))
    return dict(cpu=total_cpu / ncpu, title_cpu=title_cpu, wine_cpu=wine_cpu, ncpu=ncpu,
                ram_used=used, ram_total=memsize, title_rss=title_rss,
                swap_mb=float(swap.group(1)) if swap else 0.0,
                gpu=int(gpu.group(1)) if gpu else None,
                gpu_mem=int(gpu_mem.group(1)) if gpu_mem else None)


def gb(n):
    return f"{n / 2**30:5.1f} GB"


def render():
    now = time.time()
    out = []
    line = "─" * WIDTH
    out.append(f" SCARLETTMAC HUD{time.strftime('%H:%M:%S'):>{WIDTH - 16}}")
    out.append(line)
    r = run_state()
    if r is None:
        out.append(" no runs yet")
    else:
        elapsed = (now if r["live"] else r["mtime"]) - r["started"]
        state = "RUNNING" if r["live"] else ("CRASHED" if r["crash"] else "ENDED")
        out.append(f" run   {r['tag'][:34]:<34} {state:>8} {int(elapsed)//60:3d}m{int(elapsed)%60:02d}s")
        if r["planned"]:
            out.append(f" time  {bar(elapsed / r['planned'])} {int(elapsed):4d}/{r['planned']} s")
        s = r["samples"][-1] if r["samples"] else None
        if r["first_world"]:
            load_s = r["first_world"] - r["started"]
            out.append(f" load  {bar(1.0)} world at {int(load_s)} s")
        else:
            phase = "loading" if s else "starting"
            out.append(f" load  {bar(elapsed / TYPICAL_WORLD_S)} {phase} ~{min(99, int(100 * elapsed / TYPICAL_WORLD_S))}%")
        if s:
            age = now - s["t"] if s["t"] else 0
            stale = f"  (last sample {int(age)} s ago)" if r["live"] and age > 15 else ""
            out.append(line)
            out.append(f" fps   {s['fps']:6.1f}   frame {s['ms']:9.1f} ms   draws {s['draws']:6d}{stale}")
            iv = r["intervals"][-1] if r["intervals"] else None
            out.append(f" pace  interval {iv if iv is not None else 0:6.1f} ms   frame# {r['frame'] or 0:<7d} dropped {s['dropped']}")
            out.append(f" cost  copies {s['copy_ms']:8.0f} ms {s['copy_mb']:7.1f} MB   GPU {s['gpu']:8.0f} ms")
            out.append(f"       textures {s['tex_ms']:6.0f} ms {s['tex_mb']:7.1f} MB   prep {s['prep']:7.0f} ms")
            fps = [x["fps"] for x in r["samples"]]
            world = [x["fps"] for x in r["samples"] if x["draws"] >= WORLD_DRAWS]
            out.append(f" fps   {spark(fps, WIDTH - 8)}")
            if world:
                world_sorted = sorted(world)
                out.append(f" world median {world_sorted[len(world_sorted)//2]:.1f} fps   min {world_sorted[0]:.1f}   max {world_sorted[-1]:.1f}   ({len(world)} samples)")
        if r["crash"]:
            out.append(f" CRASH {r['crash']}")
    out.append(line)
    q = [t for t, _, pid in launchers() if pid != lock_owner()]
    owners = {t for t, _, pid in launchers() if pid == lock_owner()}
    out.append(f" lock  {', '.join(owners) or ('held' if lock_owner() else 'free')}")
    out.append(f" queue {', '.join(sorted(set(q) - owners)) or 'empty'}")
    out.append(line)
    m = system()
    out.append(f" CPU   {bar(m['cpu'] / 100, 24)} {m['cpu']:5.1f}%  title {m['title_cpu']:.0f}%  wineserver {m['wine_cpu']:.0f}%")
    out.append(f" RAM   {bar(m['ram_used'] / m['ram_total'], 24)} {gb(m['ram_used'])} / {gb(m['ram_total']).strip()}")
    out.append(f"       title {gb(m['title_rss']).strip()}   swap {m['swap_mb'] / 1024:.1f} GB")
    if m["gpu"] is not None:
        gm = f"   GPU mem {gb(m['gpu_mem']).strip()}" if m["gpu_mem"] else ""
        out.append(f" GPU   {bar(m['gpu'] / 100, 24)} {m['gpu']:3d}%{gm}")
    return "\n".join(out)


def main():
    if "--once" in sys.argv:
        print(render())
        return
    try:
        sys.stdout.write("\x1b[?25l")
        while True:
            frame = render()
            sys.stdout.write("\x1b[H\x1b[2J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h\n")


if __name__ == "__main__":
    main()
