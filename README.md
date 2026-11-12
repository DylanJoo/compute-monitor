# compute-monitor

See which GPUs on the Slurm cluster you can use **right now**, and **when** more will free up.
A cron job on a login node rebuilds the dashboard every 10 minutes and publishes it to
**GitHub Pages**: `https://dylanjoo.github.io/compute-monitor/`.

- **Live**: free GPUs per type and node, counting only GPUs on healthy nodes that also have
  spare CPU/RAM, filtered by *your* QOS limits (e.g. `4gpu_tier` = max 4 GPUs, no H100/H200).
- **Forecast**: when 1/2/4/8 GPUs of each type will be free on one node, assuming running jobs
  use their full time limit (worst case; most finish well before).
- **Patterns**: typical free GPUs by weekday × hour, and the last 14 days. History is backfilled
  from `sacct` on first run, then refined by snapshots every run.

No dependencies beyond Python 3.9+ and the Slurm CLI tools.

## Usage

```bash
python3 -m gpumon status          # terminal summary (add --nodes for per-node detail)
python3 -m gpumon publish         # rebuild and force-push the dashboard to the gh-pages branch
python3 -m gpumon serve           # optional local dashboard on :8765
```

## Auto-publishing (how the GitHub page stays fresh)

`scripts/cron_update.sh` collects, rebuilds and publishes. Installed on a login node with:

```bash
( crontab -l 2>/dev/null; echo "*/10 * * * * $PWD/scripts/cron_update.sh" ) | crontab -
```

GitHub side, once: **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `(root)`**.

- `gh-pages` is force-pushed as a single commit each run, so it never accumulates history,
  and `main` is never touched by the automation.
- The published `data.json` contains **only GPU counts per type/node**: no usernames or job IDs.
  Node hostnames, GPU models and your QOS tier *are* visible; the repo is public, so is the page.
- Cron runs on one specific login node (`crontab` is per-host); if that node is rebuilt, reinstall.
  Logs: `data/cron.log`. The page shows an amber dot when data is more than 30 min old.

## Configuration

Environment variables (see `gpumon/config.py`): `GPUMON_CPUS_PER_GPU` (2), `GPUMON_MEM_GB_PER_GPU` (16),
`GPUMON_HISTORY_DAYS` (28), `GPUMON_FORECAST_HOURS` (72), `GPUMON_SITE_TITLE`.

## Caveats

- Slurm's own `sbatch --test-only` estimate is not used: on this cluster it reports ~1.5 days
  even for idle nodes.
- History before monitoring started is rebuilt from job accounting against *today's* online
  capacity, so past node outages aren't reflected until live snapshots cover that period.
- Forecasts ignore queued jobs that may grab GPUs first (the dashboard shows how many are waiting).
