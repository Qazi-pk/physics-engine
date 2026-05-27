from __future__ import annotations

import numpy as np


class Dataset:
    """
    Standard container for physical datasets.
    """

    def __init__(self, variables: list[str], data: dict[str, np.ndarray], time: np.ndarray | None = None):
        """
        variables : list[str]
        data : dict[str, array]
        time : optional time array
        """
        self.variables = list(variables)
        self.data = dict(data)
        self.time = time

    def get(self, name: str):
        return self.data.get(name)

    def summary(self) -> dict:
        samples = 0
        if self.data:
            first_key = next(iter(self.data.keys()))
            samples = len(np.asarray(self.data[first_key]))

        return {
            "variables": self.variables,
            "samples": samples,
        }

    def to_numpy(self) -> np.ndarray:
        return np.column_stack([np.asarray(self.data[v]) for v in self.variables])
