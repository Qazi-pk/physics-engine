"""
Cache manager for benchmark experiment results.

Stores one JSON artifact per deterministic experiment_id.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class CacheManager:
    """Manages persistent experiment result cache."""

    def __init__(self, cache_dir: Path | str = "results/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, experiment_id: str) -> Path:
        """Return cache file path for experiment id."""
        return self.cache_dir / f"{experiment_id}.json"

    def exists(self, experiment_id: str) -> bool:
        """Check whether cache entry exists."""
        return self.path_for(experiment_id).exists()

    def load(self, experiment_id: str) -> Dict[str, Any]:
        """Load cached entry by experiment id."""
        with open(self.path_for(experiment_id), "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, experiment_id: str, result: Dict[str, Any]) -> Path:
        """Save cache entry by experiment id."""
        payload = dict(result)
        payload.setdefault("experiment_id", experiment_id)
        payload.setdefault("cached_at", datetime.now().isoformat())
        path = self.path_for(experiment_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def invalidate(self, experiment_id: str) -> bool:
        """Delete cache entry if present."""
        path = self.path_for(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_ids(self) -> list[str]:
        """List cached experiment IDs."""
        return sorted(path.stem for path in self.cache_dir.glob("*.json"))

    def stats(self) -> Dict[str, Optional[float]]:
        """Return cache usage statistics."""
        files = list(self.cache_dir.glob("*.json"))
        size_bytes = sum(file.stat().st_size for file in files)
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(files),
            "size_mb": size_bytes / (1024 * 1024),
        }


__all__ = ["CacheManager"]
