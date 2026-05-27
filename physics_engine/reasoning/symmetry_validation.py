from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from physics_engine.symmetry import ConservationFinder, SymmetryDetector


@dataclass(frozen=True)
class SymmetryValidationResult:
    rotational_symmetry: bool
    rotational_symmetry_score: float
    rotational_symmetry_coverage: float
    angular_momentum_conserved: bool
    angular_momentum_score: float
    confidence: float

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "rotational_symmetry": self.rotational_symmetry,
            "rotational_symmetry_score": self.rotational_symmetry_score,
            "rotational_symmetry_coverage": self.rotational_symmetry_coverage,
            "angular_momentum_conserved": self.angular_momentum_conserved,
            "angular_momentum_score": self.angular_momentum_score,
            "confidence": self.confidence,
        }


class SymmetryValidator:
    def __init__(self, symmetry_tolerance: float = 0.15, conservation_tolerance: float = 1e-3) -> None:
        self.detector = SymmetryDetector(tolerance=symmetry_tolerance)
        self.conservation = ConservationFinder(tolerance=conservation_tolerance)

    @staticmethod
    def _extract_array(data: Mapping[str, object] | pd.DataFrame, key: str) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if key not in data.columns:
                raise KeyError(f"Missing required column: {key}")
            return np.asarray(data[key].to_numpy(dtype=float), dtype=float).reshape(-1)

        if key not in data:
            raise KeyError(f"Missing required key: {key}")
        return np.asarray(data[key], dtype=float).reshape(-1)

    def analyze_orbit_system(
        self,
        data: Mapping[str, object] | pd.DataFrame,
        dt: float = 1.0,
        mass: float = 1.0,
    ) -> SymmetryValidationResult:
        x = self._extract_array(data, "x")
        y = self._extract_array(data, "y")
        vx = self._extract_array(data, "vx")
        vy = self._extract_array(data, "vy")
        ax = self._extract_array(data, "ax")
        ay = self._extract_array(data, "ay")

        rotational = self.detector.detect_rotational_symmetry(x=x, y=y, ax=ax, ay=ay)
        angular = self.conservation.detect_angular_momentum(x=x, y=y, vx=vx, vy=vy, dt=dt, mass=mass)

        symmetry_conf = max(0.0, 1.0 - rotational.score)
        conservation_conf = max(0.0, 1.0 - angular.score)
        confidence = float(np.clip(0.5 * symmetry_conf + 0.5 * conservation_conf, 0.0, 1.0))

        return SymmetryValidationResult(
            rotational_symmetry=rotational.detected,
            rotational_symmetry_score=rotational.score,
            rotational_symmetry_coverage=rotational.coverage,
            angular_momentum_conserved=angular.conserved,
            angular_momentum_score=angular.score,
            confidence=confidence,
        )


def validate_orbit_symmetry(
    data: Mapping[str, object] | pd.DataFrame,
    dt: float = 1.0,
    mass: float = 1.0,
    symmetry_tolerance: float = 0.15,
    conservation_tolerance: float = 1e-3,
) -> SymmetryValidationResult:
    validator = SymmetryValidator(
        symmetry_tolerance=symmetry_tolerance,
        conservation_tolerance=conservation_tolerance,
    )
    return validator.analyze_orbit_system(data=data, dt=dt, mass=mass)
