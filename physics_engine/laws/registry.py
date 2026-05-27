from typing import Iterable, List


class LawRegistry:
    def __init__(self) -> None:
        self._laws: List = []

    def register(self, law) -> None:
        self._laws.append(law)

    def extend(self, laws: Iterable) -> None:
        for law in laws:
            self.register(law)

    def list(self):
        return list(self._laws)

    def applicable(self, metadata):
        applicable_laws = []
        for law in self._laws:
            if hasattr(law, "applicable") and law.applicable(metadata):
                applicable_laws.append(law)
        return applicable_laws


registry = LawRegistry()
