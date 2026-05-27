from __future__ import annotations

import json
from pathlib import Path

from .models import KnowledgeLaw


DEFAULT_DOMAINS = ("mechanics", "electromagnetism", "thermodynamics", "quantum")


def _knowledge_dir() -> Path:
    return Path(__file__).resolve().parent


def load_domain(domain: str) -> list[KnowledgeLaw]:
    path = _knowledge_dir() / f"{domain}.json"
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    return [KnowledgeLaw.from_dict(item, default_domain=domain) for item in raw if isinstance(item, dict)]


def load_knowledge_base(domains: tuple[str, ...] | None = None) -> list[KnowledgeLaw]:
    active_domains = domains or DEFAULT_DOMAINS
    entries: list[KnowledgeLaw] = []
    for domain in active_domains:
        entries.extend(load_domain(domain))
    return entries
