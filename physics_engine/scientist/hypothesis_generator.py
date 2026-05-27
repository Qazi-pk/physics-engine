from __future__ import annotations

from physics_engine.knowledge import search_laws


def generate_guided_hypotheses(
    target_var: str,
    question: str = "",
    domain: str | None = None,
    max_candidates: int = 5,
) -> list[str]:
    query = f"{question} target {target_var}".strip()
    matched_laws = search_laws(query=query, domain=domain, top_k=max_candidates)

    hypotheses: list[str] = []
    for law in matched_laws:
        hypotheses.append(f"{target_var} may follow {law.law}: {law.equation}")

    if not hypotheses:
        hypotheses.append(f"{target_var} likely depends on low-order combinations of observed variables")

    return hypotheses[:max_candidates]
