"""Thin wrappers around Slurm CLI tools, returning plain dicts."""
import getpass
import re
import subprocess
from datetime import datetime

# Node states in which Slurm will not start new work.
UNAVAILABLE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "DRAINING", "FAIL", "FAILING", "MAINT",
    "RESERVED", "NOT_RESPONDING", "INVALID", "UNKNOWN", "FUTURE",
    "REBOOT_REQUESTED", "REBOOT_ISSUED", "POWER_DOWN", "POWERING_DOWN",
}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_mem_mb(s):
    """'584G' -> 598016, '1000000M' -> 1000000, '2T' -> ..."""
    m = re.fullmatch(r"([\d.]+)([KMGTP]?)", s)
    if not m:
        return 0
    val, unit = float(m.group(1)), m.group(2) or "M"
    return int(val * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}[unit])


def parse_tres(s):
    """'cpu=8,mem=96G,gres/gpu=1,gres/gpu:l40s=1' -> {'cpu': 8, 'mem_mb': 98304, 'gpu': {'l40s': 1}, 'gpu_total': 1}"""
    out = {"cpu": 0, "mem_mb": 0, "gpu": {}, "gpu_total": 0}
    if not s or s in ("(null)", "N/A"):
        return out
    for part in s.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k == "cpu":
            out["cpu"] = int(v)
        elif k == "mem":
            out["mem_mb"] = parse_mem_mb(v)
        elif k == "gres/gpu":
            out["gpu_total"] = int(v)
        elif k.startswith("gres/gpu:"):
            out["gpu"][k.split(":", 1)[1]] = int(v)
    return out


def parse_time(s):
    if not s or s in ("Unknown", "None", "N/A", "(null)"):
        return None
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp())


def parse_duration(s):
    """Slurm duration '[D-]HH:MM:SS' / 'MM:SS' -> seconds. UNLIMITED/invalid -> None."""
    if not s or s in ("UNLIMITED", "Partition_Limit", "INVALID", "NOT_SET"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def _kv_line(line):
    """Parse a `scontrol -o` line into a dict (values never contain spaces for the keys we use)."""
    d = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def gpu_nodes():
    """All nodes that have GPUs configured, with capacity, allocation and availability."""
    nodes = []
    for line in run(["scontrol", "-o", "show", "node"]).splitlines():
        d = _kv_line(line)
        cfg = parse_tres(d.get("CfgTRES", ""))
        if not cfg["gpu"]:
            continue
        alloc = parse_tres(d.get("AllocTRES", ""))
        state = d.get("State", "UNKNOWN")
        flags = set(re.split(r"[+]", state.rstrip("*$~#!%@^-")))
        if state.endswith("*"):
            flags.add("NOT_RESPONDING")
        gpu_type, gpu_total = next(iter(cfg["gpu"].items()))
        cpus_total = int(d.get("CPUEfctv", d.get("CPUTot", 0)))
        nodes.append({
            "name": d["NodeName"],
            "partitions": d.get("Partitions", ""),
            "state": state,
            "available": not (flags & UNAVAILABLE_STATES),
            "gpu_type": gpu_type,
            "gpu_total": gpu_total,
            "gpu_alloc": alloc["gpu"].get(gpu_type, alloc["gpu_total"]),
            "cpu_total": cpus_total,
            "cpu_alloc": int(d.get("CPUAlloc", 0)),
            "mem_total_mb": int(d.get("RealMemory", 0)),
            "mem_alloc_mb": int(d.get("AllocMem", 0)),
        })
    return nodes


def running_gpu_jobs():
    """Running jobs holding GPUs: node, gpu count, cpu/mem, start/end (end = start + time limit)."""
    fmt = "JobID:30,UserName:20,NodeList:200,tres-alloc:200,StartTime:22,EndTime:22,NumNodes:8"
    out = run(["squeue", "-h", "-t", "R", "-O", fmt])
    jobs = []
    for line in out.splitlines():
        f = line.split()
        if len(f) < 7:
            continue
        jobid, user, nodelist, tres, start, end, nnodes = f[:7]
        t = parse_tres(tres)
        if not t["gpu_total"]:
            continue
        nn = max(int(nnodes), 1)
        hosts = [nodelist] if nn == 1 else run(["scontrol", "show", "hostnames", nodelist]).split()
        for h in hosts:  # split multi-node jobs evenly across their nodes
            jobs.append({
                "jobid": jobid, "user": user, "node": h,
                "gpus": t["gpu_total"] // nn, "cpus": t["cpu"] // nn, "mem_mb": t["mem_mb"] // nn,
                "start": parse_time(start), "end": parse_time(end),
            })
    return jobs


def pending_gpu_demand():
    """Pending GPU requests per type, split by whether they are actually competing for resources."""
    out = run(["squeue", "-h", "-t", "PD", "-O", "tres-alloc:200,Reason:40"])
    demand = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) < 2:
            continue
        t = parse_tres(f[0])
        if not t["gpu_total"]:
            continue
        gtype = next(iter(t["gpu"]), "any")
        reason = f[1]
        d = demand.setdefault(gtype, {"competing": 0, "blocked": 0})
        # Jobs waiting on Resources/Priority are ahead of you in line; held / dependency /
        # per-user-limit jobs are not going to take a free GPU right now.
        key = "competing" if reason in ("Resources", "Priority", "None") else "blocked"
        d[key] += t["gpu_total"]
    return demand


def my_limits(user=None):
    """Per-user GPU caps from the user's QOS: {'total': 4, 'per_type': {'h100': 0, ...}} (None = no cap)."""
    user = user or getpass.getuser()
    lim = {"user": user, "qos": None, "total": None, "per_type": {}}
    try:
        rows = run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user}",
                    "format=qos,defaultqos,maxtres"]).splitlines()
        if not rows:
            return lim
        qos_list, default_qos, maxtres = (rows[0].split("|") + ["", "", ""])[:3]
        qos = default_qos or qos_list.split(",")[0]
        lim["qos"] = qos or None
        caps = maxtres
        if qos:
            q = run(["sacctmgr", "-n", "-P", "show", "qos", qos, "format=maxtrespu"]).strip()
            caps = ",".join(x for x in (caps, q) if x)
        for part in filter(None, caps.split(",")):
            k, _, v = part.partition("=")
            if k == "gres/gpu":
                lim["total"] = int(v)
            elif k.startswith("gres/gpu:"):
                lim["per_type"][k.split(":", 1)[1]] = int(v)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return lim


def sacct_gpu_jobs(since_ts, until_ts):
    """GPU jobs that were running at any point in [since, until]."""
    fmt = "JobID,AllocTRES,Start,End,Elapsed,Timelimit,State"
    s = datetime.fromtimestamp(since_ts).strftime("%Y-%m-%dT%H:%M:%S")
    e = datetime.fromtimestamp(until_ts).strftime("%Y-%m-%dT%H:%M:%S")
    out = run(["sacct", "-a", "-X", "-n", "-P", "-s", "R", "-S", s, "-E", e, "-o", fmt])
    jobs = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 7:
            continue
        jobid, tres, start, end, elapsed, limit, state = f[:7]
        t = parse_tres(tres)
        st = parse_time(start)
        if not t["gpu_total"] or st is None:
            continue
        jobs.append({
            "jobid": jobid,
            "gpu_type": next(iter(t["gpu"]), "unknown"),
            "gpus": t["gpu_total"],
            "start": st,
            "end": parse_time(end),  # None while still running
            "elapsed_s": parse_duration(elapsed),
            "limit_s": parse_duration(limit),
            "state": state.split()[0],
        })
    return jobs
