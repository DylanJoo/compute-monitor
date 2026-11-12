"""SQLite persistence: periodic per-node snapshots + sacct job history."""
import sqlite3

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts INTEGER NOT NULL, node TEXT NOT NULL, gpu_type TEXT NOT NULL,
    total INTEGER, alloc INTEGER, usable INTEGER, available INTEGER,
    PRIMARY KEY (ts, node)
);
CREATE INDEX IF NOT EXISTS snapshots_ts ON snapshots(ts);
CREATE TABLE IF NOT EXISTS jobs (
    jobid TEXT PRIMARY KEY, gpu_type TEXT, gpus INTEGER,
    start INTEGER, end INTEGER, elapsed_s INTEGER, limit_s INTEGER, state TEXT
);
CREATE INDEX IF NOT EXISTS jobs_start ON jobs(start);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(config.DB_PATH))
    db.executescript(SCHEMA)
    return db


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))


def save_snapshot(db, ts, nodes):
    db.executemany(
        "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?)",
        [(ts, n["name"], n["gpu_type"], n["gpu_total"], n["gpu_alloc"], n["usable"], int(n["available"]))
         for n in nodes],
    )


def upsert_jobs(db, jobs):
    db.executemany(
        "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)",
        [(j["jobid"], j["gpu_type"], j["gpus"], j["start"], j["end"], j["elapsed_s"], j["limit_s"], j["state"])
         for j in jobs],
    )


def prune(db, before_ts):
    db.execute("DELETE FROM snapshots WHERE ts < ?", (before_ts,))
    db.execute("DELETE FROM jobs WHERE end IS NOT NULL AND end < ?", (before_ts,))
