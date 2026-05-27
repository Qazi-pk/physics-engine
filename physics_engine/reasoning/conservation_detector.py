from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def detect_conservation(quantity: np.ndarray | list[float], time: np.ndarray | list[float] | None = None) -> float:
    """Return mean absolute time derivative for a candidate conserved quantity.

    Lower score is better; values near zero indicate likely conservation.
    """
    q = np.asarray(quantity, dtype=float).reshape(-1)
    if q.size < 2:
        return float("inf")

    if time is None:
        dqdt = np.gradient(q)
    else:
        t = np.asarray(time, dtype=float).reshape(-1)
        if t.size != q.size:
            return float("inf")
        dqdt = np.gradient(q, t)

    return float(np.mean(np.abs(dqdt)))


@dataclass(frozen=True)
class ConservationDetectionResult:
    detected: bool
    quantity_name: str
    expression: str
    score: float
    threshold: float

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "detected": self.detected,
            "quantity_name": self.quantity_name,
            "expression": self.expression,
            "score": self.score,
            "threshold": self.threshold,
        }


def detect_conserved_quantities(
    data: Mapping[str, object] | pd.DataFrame,
    *,
    threshold: float = 1e-3,
    time_col: str | None = None,
) -> ConservationDetectionResult | None:
    """Detect conservation from simple physics-informed candidate quantities.

    Candidate set is intentionally minimal and assumption-light.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    if df.empty:
        return None

    time: np.ndarray | None = None
    if time_col and time_col in df.columns:
        time = df[time_col].to_numpy(dtype=float)
    elif "t" in df.columns:
        time = df["t"].to_numpy(dtype=float)
    elif "time" in df.columns:
        time = df["time"].to_numpy(dtype=float)

    candidates: list[tuple[str, str, np.ndarray]] = []

    if {"x", "v"}.issubset(df.columns):
        x = df["x"].to_numpy(dtype=float)
        v = df["v"].to_numpy(dtype=float)
        e = 0.5 * (v**2) + 0.5 * (x**2)
        candidates.append(("energy", "0.5*v^2 + 0.5*x^2", e))

    if {"q", "dqdt"}.issubset(df.columns):
        q = df["q"].to_numpy(dtype=float)
        dqdt = df["dqdt"].to_numpy(dtype=float)
        e = 0.5 * (dqdt**2) + 0.5 * (q**2)
        candidates.append(("energy", "0.5*dqdt^2 + 0.5*q^2", e))

    if {"theta", "omega"}.issubset(df.columns):
        theta = df["theta"].to_numpy(dtype=float)
        omega = df["omega"].to_numpy(dtype=float)
        e = 0.5 * (omega**2) + (1.0 - np.cos(theta))
        candidates.append(("energy", "0.5*omega^2 + (1 - cos(theta))", e))

    if {"x", "y", "vx", "vy"}.issubset(df.columns):
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        vx = df["vx"].to_numpy(dtype=float)
        vy = df["vy"].to_numpy(dtype=float)
        lz = x * vy - y * vx
        candidates.append(("angular_momentum", "x*vy - y*vx", lz))

    if not candidates:
        return None

    best_name = ""
    best_expr = ""
    best_score = float("inf")
    for name, expr, values in candidates:
        score = detect_conservation(values, time=time)
        if score < best_score:
            best_name = name
            best_expr = expr
            best_score = score

    return ConservationDetectionResult(
        detected=bool(best_score <= float(threshold)),
        quantity_name=best_name,
        expression=best_expr,
        score=float(best_score),
        threshold=float(threshold),
    )


__all__ = [
    "ConservationDetectionResult",
    "detect_conservation",
    "detect_conserved_quantities",
]
