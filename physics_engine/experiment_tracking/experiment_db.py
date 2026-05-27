"""
SQLite-backed experiment metadata index for large-scale benchmarking.

Stores compact metadata for each experiment while large artifacts remain on disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentDB:
    """Lightweight SQLite metadata index for benchmark experiments."""

    def __init__(self, path: str | Path = "results/experiments.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE,
                run_id INTEGER,
                dataset TEXT,
                algorithm TEXT,
                noise REAL,
                dataset_size INTEGER,
                seed INTEGER,
                status TEXT,
                success INTEGER,
                error REAL,
                runtime REAL,
                from_cache INTEGER,
                artifact_dir TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_algorithm ON experiments(algorithm)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_noise ON experiments(noise)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def has_experiment(self, experiment_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
            (experiment_id,),
        ).fetchone()
        return row is not None

    def insert_or_update(self, record: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO experiments(
                experiment_id, run_id, dataset, algorithm, noise, dataset_size, seed,
                status, success, error, runtime, from_cache, artifact_dir
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                run_id=excluded.run_id,
                dataset=excluded.dataset,
                algorithm=excluded.algorithm,
                noise=excluded.noise,
                dataset_size=excluded.dataset_size,
                seed=excluded.seed,
                status=excluded.status,
                success=excluded.success,
                error=excluded.error,
                runtime=excluded.runtime,
                from_cache=excluded.from_cache,
                artifact_dir=excluded.artifact_dir
            """,
            (
                record.get("experiment_id"),
                int(record.get("run_id", 0) or 0),
                str(record.get("dataset", "")),
                str(record.get("algorithm", "")),
                float(record.get("noise", record.get("noise_level", 0.0)) or 0.0),
                int(record.get("dataset_size", 0) or 0),
                int(record.get("seed", 0) or 0),
                str(record.get("status", "unknown")),
                1 if bool(record.get("success", False)) else 0,
                float(record.get("error", 0.0) or 0.0),
                float(record.get("runtime", record.get("runtime_seconds", 0.0)) or 0.0),
                1 if bool(record.get("from_cache", False)) else 0,
                str(record.get("artifact_dir", "")),
            ),
        )
        self.conn.commit()

    def fetch(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    def fetch_all(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM experiments ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]

    def aggregate_success_by_algorithm(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT algorithm,
                   COUNT(*) AS total,
                   AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate,
                   AVG(runtime) AS avg_runtime,
                   AVG(error) AS avg_error
            FROM experiments
            GROUP BY algorithm
            ORDER BY algorithm ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful,
              SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
              SUM(CASE WHEN from_cache = 1 THEN 1 ELSE 0 END) AS cache_hits
            FROM experiments
            """
        ).fetchone()
        total = int(row["total"] or 0)
        successful = int(row["successful"] or 0)
        failed = int(row["failed"] or 0)
        cache_hits = int(row["cache_hits"] or 0)
        return {
            "db_path": str(self.path),
            "total": total,
            "successful": successful,
            "failed": failed,
            "cache_hits": cache_hits,
            "success_rate": (successful / total) if total else 0.0,
        }
