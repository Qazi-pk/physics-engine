from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FRANKA_COMPONENTS: tuple[str, ...] = ("M11", "M22", "M33", "M44", "M55", "M66", "M77")
FRANKA_VARIANTS: tuple[str, ...] = ("baseline", "payload", "rotor4", "link5mass")


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "benchmarks" / "franka_mass"


def _dataset_filename(component: str, variant: str) -> str:
    if variant == "baseline":
        return f"franka_{component}.csv"
    return f"franka_{component}_{variant}.csv"


def generate_franka_mass_dataset(
    *,
    component: str,
    variant: str = "baseline",
    num_samples: int = 200,
    noise_std: float = 0.0,
    seed: int = 123,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load a pre-generated Franka mass-diagonal dataset and sample rows.

    The source CSVs are generated once by `benchmarks/franka_mass/gen_franka_mass_data.py`.
    This loader keeps run-time cost low for benchmark sweeps.
    """

    if component not in FRANKA_COMPONENTS:
        raise ValueError(f"Unsupported Franka component: {component}")
    if variant not in FRANKA_VARIANTS:
        raise ValueError(f"Unsupported Franka variant: {variant}")

    root = Path(data_root) if data_root is not None else _default_data_root()
    csv_path = root / _dataset_filename(component=component, variant=variant)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Franka dataset not found: {csv_path}. "
            "Run benchmarks/franka_mass/gen_franka_mass_data.py first."
        )

    df = pd.read_csv(csv_path)

    expected_inputs = [f"q{i}" for i in range(1, 8)]
    expected_columns = expected_inputs + [component]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {csv_path} is missing required columns: {missing}")

    rng = np.random.default_rng(seed)
    replace = len(df) < num_samples
    sample_idx = rng.choice(len(df), size=num_samples, replace=replace)
    sampled = df.iloc[sample_idx][expected_columns].reset_index(drop=True)

    if noise_std > 0:
        sampled = sampled.copy()
        sampled[component] = sampled[component] + rng.normal(0.0, noise_std, size=num_samples)

    return sampled
