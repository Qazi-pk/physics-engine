from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeLaw:
    law: str
    equation: str
    domain: str
    variables: tuple[str, ...]
    description: str = ""

    @staticmethod
    def from_dict(item: dict, default_domain: str = "general") -> "KnowledgeLaw":
        return KnowledgeLaw(
            law=str(item.get("law", "")).strip(),
            equation=str(item.get("equation", "")).strip(),
            domain=str(item.get("domain", default_domain)).strip() or default_domain,
            variables=tuple(str(v).strip() for v in item.get("variables", []) if str(v).strip()),
            description=str(item.get("description", "")).strip(),
        )

    def to_dict(self) -> dict:
        return {
            "law": self.law,
            "equation": self.equation,
            "domain": self.domain,
            "variables": list(self.variables),
            "description": self.description,
        }
