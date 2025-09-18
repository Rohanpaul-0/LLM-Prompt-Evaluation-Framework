from __future__ import annotations
import sqlite3, json, time, os, subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY,
  created_ts REAL,
  config_json TEXT,
  git_commit TEXT,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS metrics(
  run_id TEXT,
  record_id TEXT,
  model TEXT,
  metric TEXT,
  value REAL
);
"""

@dataclass
class Run:
    run_id: str
    created_ts: float
    config_json: str
    git_commit: str | None = None
    notes: str | None = None

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

class Registry:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def create_run(self, config: Dict[str, Any], notes: str | None = None) -> Run:
        run_id = str(int(time.time()*1000))
        r = Run(run_id=run_id, created_ts=time.time(), config_json=json.dumps(config), git_commit=_git_commit(), notes=notes)
        self._conn.execute("INSERT INTO runs(run_id,created_ts,config_json,git_commit,notes) VALUES(?,?,?,?,?)",
                           (r.run_id, r.created_ts, r.config_json, r.git_commit, r.notes))
        self._conn.commit()
        return r

    def log_metric(self, run_id: str, record_id: str, model: str, metric: str, value: float):
        self._conn.execute("INSERT INTO metrics(run_id,record_id,model,metric,value) VALUES(?,?,?,?,?)",
                           (run_id, str(record_id), str(model), metric, float(value)))
        self._conn.commit()

    def list_runs(self):
        cur = self._conn.execute("SELECT run_id, created_ts, git_commit, notes FROM runs ORDER BY created_ts DESC")
        return cur.fetchall()

    def get_run(self, run_id: str):
        cur = self._conn.execute("SELECT run_id, created_ts, config_json, git_commit, notes FROM runs WHERE run_id=?",(run_id,))
        return cur.fetchone()
