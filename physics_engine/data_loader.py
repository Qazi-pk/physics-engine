"""
Data loader utilities for PIR Engine.

Provides convenient access to example datasets bundled with the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Get the data directory path
_DATA_DIR = Path(__file__).parent / "data"

# Available example datasets
AVAILABLE_DATASETS = {
    "pendulum": "Simple pendulum trajectory (x, y, vx, vy vs time)",
    "robot_joint": "Robot joint angles, velocities, and torques",
    "orbit": "Orbital motion data (position and velocity)",
    "spring_mass": "Spring-mass system with forces",
}


def load_example(name: str) -> pd.DataFrame:
    """
    Load an example dataset bundled with PIR Engine.
    
    Args:
        name: Name of the dataset (without .csv extension)
              Available: 'pendulum', 'robot_joint', 'orbit', 'spring_mass'
    
    Returns:
        DataFrame containing the example data
    
    Raises:
        ValueError: If dataset name is not recognized
        FileNotFoundError: If dataset file is missing
    
    Examples:
        >>> from physics_engine.data_loader import load_example
        >>> df = load_example("pendulum")
        >>> print(df.columns)
        Index(['time', 'x', 'y', 'vx', 'vy'], dtype='object')
        
        >>> robot_data = load_example("robot_joint")
        >>> print(robot_data.shape)
        (11, 10)
    """
    if name not in AVAILABLE_DATASETS:
        available = ", ".join(f"'{k}'" for k in AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )
    
    file_path = _DATA_DIR / f"{name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Please ensure PIR Engine is properly installed."
        )
    
    return pd.read_csv(file_path)


def list_datasets() -> dict[str, str]:
    """
    List all available example datasets.
    
    Returns:
        Dictionary mapping dataset names to descriptions
    
    Examples:
        >>> from physics_engine.data_loader import list_datasets
        >>> datasets = list_datasets()
        >>> for name, desc in datasets.items():
        ...     print(f"{name}: {desc}")
        pendulum: Simple pendulum trajectory (x, y, vx, vy vs time)
        robot_joint: Robot joint angles, velocities, and torques
        ...
    """
    return AVAILABLE_DATASETS.copy()


def get_data_path(name: str) -> Path:
    """
    Get the file path to an example dataset.
    
    Args:
        name: Name of the dataset (without .csv extension)
    
    Returns:
        Path object pointing to the dataset file
    
    Raises:
        ValueError: If dataset name is not recognized
    
    Examples:
        >>> from physics_engine.data_loader import get_data_path
        >>> path = get_data_path("pendulum")
        >>> print(path.exists())
        True
    """
    if name not in AVAILABLE_DATASETS:
        available = ", ".join(f"'{k}'" for k in AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )
    
    return _DATA_DIR / f"{name}.csv"


__all__ = [
    "load_example",
    "list_datasets",
    "get_data_path",
    "AVAILABLE_DATASETS",
]
