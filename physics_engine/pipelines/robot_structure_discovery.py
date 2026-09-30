from __future__ import annotations

import numpy as np

from physics_engine.core.dataset import Dataset
from physics_engine.discovery.feature_library import FeatureLibrary
from physics_engine.discovery.structure_discovery import StructureDiscovery


class RobotStructureDiscovery:
    def __init__(self, threshold: float = 0.05, library_profile: str = "basic"):
        self.library = FeatureLibrary(profile=library_profile)
        self.discovery = StructureDiscovery(threshold=threshold)

    def run(self, dataset: Dataset) -> list[tuple[float, str]]:
        torque = dataset.get("torque")
        if torque is None:
            torque = dataset.get("tau")
        if torque is None:
            raise ValueError("Dataset must contain torque or tau.")

        y = np.asarray(torque, dtype=float)
        X, names = self.library.build(dataset)
        return self.discovery.discover(X, y, names)
