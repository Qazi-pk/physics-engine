from __future__ import annotations

import re

from .loader import load_knowledge_base
from .models import KnowledgeLaw


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", (text or "").lower()) if len(token) > 0}


def search_laws(query: str, domain: str | None = None, top_k: int = 5) -> list[KnowledgeLaw]:
    entries = load_knowledge_base(domains=(domain,) if domain else None)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return entries[:top_k]

    scored: list[tuple[float, KnowledgeLaw]] = []
    for entry in entries:
        law_tokens = _tokenize(entry.law)
        equation_tokens = _tokenize(entry.equation)
        variable_tokens = {v.lower() for v in entry.variables}
        doc_tokens = law_tokens | equation_tokens | variable_tokens
        overlap = len(query_tokens & doc_tokens)
        if overlap <= 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def find_by_variables(variables: list[str], domain: str | None = None) -> list[KnowledgeLaw]:
    entries = load_knowledge_base(domains=(domain,) if domain else None)
    target = {item.lower().strip() for item in variables if item and item.strip()}
    if not target:
        return []

    matches: list[KnowledgeLaw] = []
    for entry in entries:
        vars_lower = {item.lower() for item in entry.variables}
        if target.issubset(vars_lower):
            matches.append(entry)
    return matches
