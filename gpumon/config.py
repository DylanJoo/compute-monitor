"""Runtime configuration. Every value can be overridden with an env var GPUMON_<NAME>."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
    v = os.environ.get("GPUMON_" + name)
    return cast(v) if v is not None else default


# Where the SQLite history lives (never published).
DB_PATH = Path(_env("DB_PATH", str(ROOT / "data" / "monitor.db")))
# Where the static site (index.html + data.json) is built.
BUILD_DIR = Path(_env("BUILD_DIR", str(ROOT / "build")))

# A free GPU only counts as "usable" if its node also has room for a modest job:
# at least this many free CPUs and this much free memory *per GPU*.
CPUS_PER_GPU = _env("CPUS_PER_GPU", 2, int)
MEM_GB_PER_GPU = _env("MEM_GB_PER_GPU", 16, int)

# How far back to backfill job history from sacct on first run.
HISTORY_DAYS = _env("HISTORY_DAYS", 28, int)
# How far ahead the forecast looks, and its resolution.
FORECAST_HOURS = _env("FORECAST_HOURS", 72, int)
FORECAST_STEP_MIN = _env("FORECAST_STEP_MIN", 15, int)
# Timeline window shown on the dashboard.
TIMELINE_DAYS = _env("TIMELINE_DAYS", 14, int)

# Title shown on the dashboard (keep it generic: the published site is public).
SITE_TITLE = _env("SITE_TITLE", "GPU availability")
