"""Turn raw Slurm state + stored history into the dashboard's data model.

The output of `build_data` is what gets written to data.json and published, so it
deliberately contains no usernames or job IDs: only GPU counts per type and node.
"""
import time
from collections import defaultdict
from datetime import datetime

from . import config

TYPE_ORDER = ["b200", "h200", "gh200", "h100", "a100", "a6000", "l40s", "l40", "a40", "v100", "rtx"]
K_VALUES = [1, 2, 4, 8]
HOUR = 3600


def type_key(t):
    return (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER), t)


def usable_gpus(available, free_gpu, free_cpu, free_mem_mb):
    """How many 1-GPU jobs of the configured modest shape could start on this node."""
    if not available or free_gpu <= 0:
        return 0
    n = free_gpu
    if config.CPUS_PER_GPU > 0:
        n = min(n, free_cpu // config.CPUS_PER_GPU)
    if config.MEM_GB_PER_GPU > 0:
        n = min(n, free_mem_mb // (config.MEM_GB_PER_GPU * 1024))
    return max(n, 0)


def annotate_nodes(nodes):
    for n in nodes:
        n["gpu_free"] = max(n["gpu_total"] - n["gpu_alloc"], 0) if n["available"] else 0
        n["usable"] = usable_gpus(n["available"], n["gpu_total"] - n["gpu_alloc"],
                                  n["cpu_total"] - n["cpu_alloc"], n["mem_total_mb"] - n["mem_alloc_mb"])
    return nodes


def _node_timeline(node, jobs):
    """[(t, usable_at_t)] at now and after each job end on this node (worst case: jobs run to limit)."""
    free_gpu = node["gpu_total"] - node["gpu_alloc"]
    free_cpu = node["cpu_total"] - node["cpu_alloc"]
    free_mem = node["mem_total_mb"] - node["mem_alloc_mb"]
    pts = [(None, usable_gpus(node["available"], free_gpu, free_cpu, free_mem))]
    for j in sorted((j for j in jobs if j["end"]), key=lambda j: j["end"]):
        free_gpu += j["gpus"]
        free_cpu += j["cpus"]
        free_mem += j["mem_mb"]
        pts.append((j["end"], usable_gpus(node["available"], free_gpu, free_cpu, free_mem)))
    return pts


def _usable_at(timeline, t, now):
    val = timeline[0][1]
    for ts, u in timeline[1:]:
        if max(ts, now) <= t:
            val = u
    return val


def forecast(nodes, jobs, now):
    by_node = defaultdict(list)
    for j in jobs:
        by_node[j["node"]].append(j)
    timelines = {n["name"]: _node_timeline(n, by_node[n["name"]]) for n in nodes}

    step = config.FORECAST_STEP_MIN * 60
    grid = [now + i * step for i in range(config.FORECAST_HOURS * 3600 // step + 1)]
    series = defaultdict(lambda: [0] * len(grid))
    earliest = defaultdict(dict)  # type -> k -> ts (None = not within known job ends)
    for n in nodes:
        tl = timelines[n["name"]]
        s = series[n["gpu_type"]]
        for i, t in enumerate(grid):
            s[i] += _usable_at(tl, t, now)
        for k in K_VALUES:
            if k > n["gpu_total"]:
                continue
            hit = next((max(ts or now, now) for ts, u in tl if u >= k), None)
            cur = earliest[n["gpu_type"]].get(k)
            if hit is not None and (cur is None or hit < cur):
                earliest[n["gpu_type"]][k] = hit
    return grid, dict(series), {t: dict(v) for t, v in earliest.items()}


def hourly_free_history(db, nodes, now):
    """Mean free GPUs per type per hour over the history window.

    Hours covered by our own snapshots use them (exact, includes node outages).
    Earlier hours are reconstructed from sacct: today's online capacity minus GPUs held by jobs.
    """
    start = (now - config.HISTORY_DAYS * 86400) // HOUR * HOUR
    end_bin = now // HOUR * HOUR
    bins = list(range(start, end_bin + 1, HOUR))
    idx = {b: i for i, b in enumerate(bins)}
    capacity = defaultdict(int)
    for n in nodes:
        if n["available"]:
            capacity[n["gpu_type"]] += n["gpu_total"]
    types = sorted({n["gpu_type"] for n in nodes}, key=type_key)

    alloc = {t: [0.0] * len(bins) for t in types}
    for gtype, gpus, st, en in db.execute(
            "SELECT gpu_type, gpus, start, end FROM jobs WHERE start < ? AND (end IS NULL OR end > ?)",
            (now, start)):
        if gtype not in alloc:
            continue
        en = min(en or now, now)
        st = max(st, start)
        b = st // HOUR * HOUR
        while b < en and b in idx:
            overlap = min(en, b + HOUR) - max(st, b)
            if overlap > 0:
                alloc[gtype][idx[b]] += gpus * overlap / HOUR
            b += HOUR

    free = {t: [max(capacity[t] - a, 0.0) for a in alloc[t]] for t in types}

    snap = defaultdict(lambda: defaultdict(list))  # bin -> type -> [free per snapshot]
    per_ts = defaultdict(lambda: defaultdict(int))
    for ts, gtype, total, al, avail in db.execute(
            "SELECT ts, gpu_type, total, alloc, available FROM snapshots WHERE ts >= ?", (start,)):
        per_ts[ts][gtype] += max(total - al, 0) if avail else 0
    for ts, by_type in per_ts.items():
        for t in types:
            snap[ts // HOUR * HOUR][t].append(by_type.get(t, 0))
    for b, by_type in snap.items():
        if b in idx:
            for t, vals in by_type.items():
                if t in free:
                    free[t][idx[b]] = sum(vals) / len(vals)
    # The current hour is partial; pin it to the live value so the timeline ends on "now".
    for t in types:
        free[t][-1] = sum(n["gpu_free"] for n in nodes if n["gpu_type"] == t)
    return bins, free, len(snap)


def weekly_pattern(bins, free):
    """Mean free GPUs and share of hours with >=1 free, by weekday x hour (cluster local time)."""
    out = {}
    for t, vals in free.items():
        acc = [[[] for _ in range(24)] for _ in range(7)]
        for b, v in zip(bins[:-1], vals[:-1]):  # skip the partial current hour
            lt = time.localtime(b)
            acc[lt.tm_wday][lt.tm_hour].append(v)
        out[t] = {
            "mean": [[round(sum(c) / len(c), 2) if c else None for c in row] for row in acc],
            "p_any": [[round(sum(1 for x in c if x >= 1) / len(c), 2) if c else None for c in row] for row in acc],
        }
    return out


def job_length_stats(db, now):
    """How much of its time limit a GPU job actually uses, weighted by GPU-time.

    Weighting by GPU-time answers "the job holding the GPU I'm waiting for - when does it
    typically end?", and ignores the flood of jobs that crash in their first minutes.
    """
    rows = db.execute(
        "SELECT gpu_type, gpus, elapsed_s, limit_s FROM jobs "
        "WHERE end IS NOT NULL AND limit_s > 0 AND elapsed_s >= 600 AND end > ?",
        (now - config.HISTORY_DAYS * 86400,)).fetchall()
    by_type = defaultdict(list)
    for t, gpus, el, lim in rows:
        by_type[t].append((min(el / lim, 1.0), gpus * el))

    def summ(rs):
        if not rs:
            return None
        rs = sorted(rs)
        total = sum(w for _, w in rs)
        acc, med = 0.0, rs[-1][0]
        for r, w in rs:
            acc += w
            if acc >= total / 2:
                med = r
                break
        return {"n": len(rs), "median_used": round(med, 3),
                "share_under_half": round(sum(w for r, w in rs if r < 0.5) / total, 3)}
    return {"all": summ([x for v in by_type.values() for x in v]),
            "by_type": {t: summ(v) for t, v in by_type.items()}}


def build_data(db, nodes, jobs, demand, limits, now):
    annotate_nodes(nodes)
    types = sorted({n["gpu_type"] for n in nodes}, key=type_key)
    node_type = {n["name"]: n["gpu_type"] for n in nodes}

    mine = defaultdict(int)
    for j in jobs:
        if j["user"] == limits.get("user"):
            mine[node_type.get(j["node"], "?")] += j["gpus"]
    my_total = sum(mine.values())

    grid, fc, earliest = forecast(nodes, jobs, now)
    bins, free_hist, snap_hours = hourly_free_history(db, nodes, now)
    tl_from = len(bins) - config.TIMELINE_DAYS * 24

    summary = []
    for t in types:
        tn = [n for n in nodes if n["gpu_type"] == t]
        cap = limits["per_type"].get(t)
        room = None
        if cap is not None or limits["total"] is not None:
            room = min(x for x in (
                None if cap is None else cap - mine[t],
                None if limits["total"] is None else limits["total"] - my_total,
            ) if x is not None)
            room = max(room, 0)
        usable = sum(n["usable"] for n in tn)
        summary.append({
            "type": t,
            "total": sum(n["gpu_total"] for n in tn),
            "offline": sum(n["gpu_total"] for n in tn if not n["available"]),
            "alloc": sum(n["gpu_alloc"] for n in tn if n["available"]),
            "free": sum(n["gpu_free"] for n in tn),
            "usable": usable,
            "my_cap": cap if cap is not None else limits["total"],
            "my_used": mine[t],
            "for_me": usable if room is None else min(usable, room),
            "allowed": room is None or room > 0 or mine[t] > 0,
            "pending": demand.get(t, {"competing": 0, "blocked": 0}),
            "earliest": {str(k): v for k, v in earliest.get(t, {}).items()},
            "forecast": fc.get(t, []),
            "timeline": [round(v, 2) for v in free_hist[t][tl_from:]],
        })

    return {
        "generated_at": now,
        "generated_local": time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(now)),
        "tz": time.strftime("%Z"),
        "utc_offset": int(datetime.now().astimezone().utcoffset().total_seconds()),
        "title": config.SITE_TITLE,
        "params": {"cpus_per_gpu": config.CPUS_PER_GPU, "mem_gb_per_gpu": config.MEM_GB_PER_GPU,
                   "history_days": config.HISTORY_DAYS, "snapshot_hours": snap_hours},
        "limits": {"qos": limits.get("qos"), "total": limits["total"], "per_type": limits["per_type"],
                   "used_total": my_total},
        "types": summary,
        "nodes": [{k: n[k] for k in ("name", "gpu_type", "gpu_total", "gpu_alloc", "gpu_free", "usable",
                                     "available", "state")} for n in sorted(nodes, key=lambda n: (type_key(n["gpu_type"]), n["name"]))],
        "forecast_t": grid,
        "timeline_t": bins[tl_from:],
        "weekly": weekly_pattern(bins, free_hist),
        "job_lengths": job_length_stats(db, now),
    }
