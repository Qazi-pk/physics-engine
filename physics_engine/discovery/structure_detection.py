from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureDetectionResult:
    structure: str
    variables: tuple[str, str] | None
    score: float
    samples: int


class StructureDetector:
    def __init__(self, tolerance: float = 0.15, min_samples: int = 8, max_bins: int = 8) -> None:
        self.tolerance = float(tolerance)
        self.min_samples = int(min_samples)
        self.max_bins = int(max_bins)

    @staticmethod
    def _cross_additive_residual(f11: float, f22: float, f12: float, f21: float) -> float:
        return abs((f11 + f22) - (f12 + f21))

    @staticmethod
    def _cross_multiplicative_residual(f11: float, f22: float, f12: float, f21: float) -> float:
        return abs((f11 * f22) - (f12 * f21))

    def _pair_scores(
        self,
        df: pd.DataFrame,
        target_col: str,
        x_col: str,
        y_col: str,
    ) -> tuple[float, float, int]:
        work = df[[x_col, y_col, target_col]].dropna().copy()
        if len(work) < self.min_samples:
            return float("inf"), float("inf"), 0

        bins = max(2, min(self.max_bins, int(np.sqrt(len(work)))))
        try:
            work["_xb"] = pd.qcut(work[x_col], q=bins, labels=False, duplicates="drop")
            work["_yb"] = pd.qcut(work[y_col], q=bins, labels=False, duplicates="drop")
        except Exception:
            return float("inf"), float("inf"), 0

        grid = work.groupby(["_xb", "_yb"], observed=True)[target_col].mean().unstack(fill_value=np.nan)
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return float("inf"), float("inf"), 0

        values = np.asarray(grid, dtype=float)
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            return float("inf"), float("inf"), 0

        scale = float(np.nanmean(np.abs(finite_vals))) + 1e-12
        mult_scale = float(np.nanmean(np.abs(finite_vals) ** 2)) + 1e-12

        add_residuals: list[float] = []
        mult_residuals: list[float] = []

        rows, cols = values.shape
        for i1 in range(rows):
            for i2 in range(i1 + 1, rows):
                for j1 in range(cols):
                    for j2 in range(j1 + 1, cols):
                        f11 = values[i1, j1]
                        f22 = values[i2, j2]
                        f12 = values[i1, j2]
                        f21 = values[i2, j1]
                        if not np.all(np.isfinite([f11, f22, f12, f21])):
                            continue
                        add_residuals.append(self._cross_additive_residual(f11, f22, f12, f21) / scale)
                        mult_residuals.append(
                            self._cross_multiplicative_residual(f11, f22, f12, f21) / mult_scale
                        )

        if not add_residuals:
            return float("inf"), float("inf"), 0

        return float(np.mean(add_residuals)), float(np.mean(mult_residuals)), len(add_residuals)

    def detect_structure(
        self,
        df: pd.DataFrame,
        target_col: str,
        variables: list[str],
    ) -> StructureDetectionResult:
        if len(variables) < 2 or target_col not in df.columns:
            return StructureDetectionResult(structure="unknown", variables=None, score=float("inf"), samples=0)

        best_structure = "unknown"
        best_pair: tuple[str, str] | None = None
        best_score = float("inf")
        best_samples = 0

        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                x_col = variables[i]
                y_col = variables[j]
                if x_col not in df.columns or y_col not in df.columns:
                    continue

                add_score, mult_score, samples = self._pair_scores(df, target_col, x_col, y_col)
                if samples < self.min_samples:
                    continue

                if add_score < best_score:
                    best_score = add_score
                    best_structure = "additive" if add_score <= self.tolerance else "unknown"
                    best_pair = (x_col, y_col)
                    best_samples = samples

                if mult_score < best_score:
                    best_score = mult_score
                    best_structure = "multiplicative" if mult_score <= self.tolerance else "unknown"
                    best_pair = (x_col, y_col)
                    best_samples = samples

        return StructureDetectionResult(
            structure=best_structure,
            variables=best_pair,
            score=float(best_score),
            samples=int(best_samples),
        )
