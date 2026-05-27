from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Theory:
    name: str
    equation: str
    confidence: float
    supporting_datasets: tuple[str, ...] = field(default_factory=tuple)
    domain: str | None = None


class TheoryManager:
    def __init__(self) -> None:
        self._theories: list[Theory] = []

    def add_theory(self, theory: Theory) -> None:
        self._theories.append(theory)

    def list_theories(self) -> list[Theory]:
        return list(self._theories)

    def top_theories(self, limit: int = 5) -> list[Theory]:
        return sorted(self._theories, key=lambda item: item.confidence, reverse=True)[:limit]

    def to_dict(self) -> list[dict]:
        return [asdict(item) for item in self._theories]
