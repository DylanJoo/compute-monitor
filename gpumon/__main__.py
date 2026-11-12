"""gpumon: see which cluster GPUs you can use now, and when more will free up.

  python3 -m gpumon status            # terminal summary (live)
  python3 -m gpumon update            # collect snapshot + history, build site into build/
  python3 -m gpumon serve [--port N]  # serve build/ locally, refreshing every 5 min
  python3 -m gpumon publish           # update + force-push build/ to the gh-pages branch
"""
import argparse
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path

from . import analyze, config, slurm, store

WEB_DIR = Path(__file__).resolve().parent / "web"


def collect(db, now):
    nodes = analyze.annotate_nodes(slurm.gpu_nodes())
    store.save_snapshot(db, now, nodes)

    last = int(store.get_meta(db, "sacct_until", 0))
    since = max(last - 3600, now - config.HISTORY_DAYS * 86400)
    store.upsert_jobs(db, slurm.sacct_gpu_jobs(since, now))
    store.set_meta(db, "sacct_until", now)
    store.prune(db, now - (config.HISTORY_DAYS + 7) * 86400)
    db.commit()
    return nodes


def update(write=True):
    now = int(time.time())
    db = store.connect()
    nodes = collect(db, now)
    data = analyze.build_data(db, nodes, slurm.running_gpu_jobs(), slurm.pending_gpu_demand(),
                              slurm.my_limits(), now)
    db.close()
    if write:
        out = config.BUILD_DIR
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "data.json.tmp"
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(out / "data.json")
        shutil.copy(WEB_DIR / "index.html", out / "index.html")
        (out / ".nojekyll").touch()
    return data


def _when(ts, now):
    if ts is None:
        return "-"
    if ts <= now + 60:
        return "now"
    mins = (ts - now) // 60
    rel = f"{mins // 60}h{mins % 60:02d}m" if mins < 48 * 60 else f"{mins // 1440}d{(mins % 1440) // 60}h"
    return f"{rel} ({datetime.fromtimestamp(ts).strftime('%a %H:%M')})"


def cmd_status(args):
    d = update(write=False)
    now = d["generated_at"]
    lim = d["limits"]
    print(f"\n  {d['title']} - {d['generated_local']}   QOS: {lim['qos']}  "
          f"(cap {lim['total'] if lim['total'] is not None else 'none'} GPUs, using {lim['used_total']})\n")
    hdr = f"  {'type':<7}{'for you':>8}{'usable':>8}{'free':>6}{'busy':>6}{'offline':>8}{'total':>7}   " \
          + "".join(f"{'next ' + str(k) + '-GPU':<22}" for k in (1, 4))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in d["types"]:
        mine = str(t["for_me"]) if t["allowed"] else "no"
        cols = "".join(f"{_when(t['earliest'].get(str(k)), now) if t['allowed'] else 'not allowed':<22}"
                       for k in (1, 4))
        print(f"  {t['type']:<7}{mine:>8}{t['usable']:>8}{t['free']:>6}{t['alloc']:>6}{t['offline']:>8}"
              f"{t['total']:>7}   {cols}")
    jl = d["job_lengths"]["all"]
    print("\n  'usable' = free GPU on a healthy node with >= "
          f"{d['params']['cpus_per_gpu']} CPUs and {d['params']['mem_gb_per_gpu']}G RAM spare per GPU.")
    print("  'next' times assume running jobs use their full time limit (worst case)"
          + (f"; in practice the median GPU job ends at {jl['median_used']:.0%} of its limit." if jl else "."))
    if args.nodes:
        print(f"\n  {'node':<12}{'type':<7}{'state':<16}{'free/total':>11}{'usable':>8}")
        for n in d["nodes"]:
            print(f"  {n['name']:<12}{n['gpu_type']:<7}{n['state']:<16}"
                  f"{str(n['gpu_free']) + '/' + str(n['gpu_total']):>11}{n['usable']:>8}")
    print()


def cmd_update(args):
    d = update()
    print(f"built {config.BUILD_DIR} at {d['generated_local']}")


def cmd_serve(args):
    def loop():
        while True:
            try:
                update()
            except Exception as e:  # keep serving the last good build
                print(f"update failed: {e}", file=sys.stderr)
            time.sleep(args.interval)
    if args.interval > 0:
        threading.Thread(target=loop, daemon=True).start()
        while not (config.BUILD_DIR / "data.json").exists():
            time.sleep(0.5)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(config.BUILD_DIR))
    srv = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving http://{args.host}:{args.port}/  (from your laptop: "
          f"ssh -N -L {args.port}:localhost:{args.port} <this-login-node>)")
    srv.serve_forever()


def cmd_publish(args):
    """Force-push build/ as a single-commit gh-pages branch, so history never grows."""
    if not args.no_update:
        update()
    remote = args.remote or subprocess.run(["git", "-C", str(config.ROOT), "remote", "get-url", "origin"],
                                           check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(config.BUILD_DIR, tmp, dirs_exist_ok=True)
        git = partial(subprocess.run, cwd=tmp, check=True, capture_output=True, text=True)
        git(["git", "init", "-q", "-b", "gh-pages"])
        git(["git", "add", "-A"])
        git(["git", "-c", "user.name=gpumon", "-c", "user.email=gpumon@users.noreply.github.com",
             "commit", "-q", "-m", f"dashboard update {time.strftime('%Y-%m-%d %H:%M %Z')}"])
        git(["git", "push", "-q", "--force", remote, "gh-pages:gh-pages"])
    print(f"published to {remote} (gh-pages)")


def main():
    p = argparse.ArgumentParser(prog="gpumon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="print what you can use now / soon")
    s.add_argument("--nodes", action="store_true", help="also list every GPU node")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("update", help="collect + build site").set_defaults(fn=cmd_update)
    s = sub.add_parser("serve", help="serve the dashboard locally")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--interval", type=int, default=300, help="seconds between refreshes (0 = never)")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("publish", help="update + push to gh-pages")
    s.add_argument("--remote", help="git remote URL (default: this repo's origin)")
    s.add_argument("--no-update", action="store_true", help="push the existing build/ as-is")
    s.set_defaults(fn=cmd_publish)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
