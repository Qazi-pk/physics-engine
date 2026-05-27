from typing import Any

from pydantic import BaseModel, Field


class ProblemInput(BaseModel):
    description: str = Field(..., description="Natural-language physics problem description.")
    known_values: dict[str, float] = Field(
        default_factory=dict,
        description="Known scalar values keyed by symbol name.",
    )


class SolutionOutput(BaseModel):
    laws: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    known_values: dict[str, Any] = Field(default_factory=dict)
    solution: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskInput(BaseModel):
    question: str = Field(..., description="Natural-language physics question.")
    dataset_path: str | None = Field(default=None, description="Optional CSV path for discovery mode.")
    target_var: str | None = Field(default=None, description="Optional target variable for discovery mode.")
    domain: str | None = Field(default=None, description="Optional knowledge domain filter.")
    run_discovery: bool = Field(default=True, description="Whether discovery routing is enabled.")
    run_scientist_loop: bool = Field(default=True, description="Whether scientist-loop routing is enabled.")
    scientist_cycles: int = Field(default=3, description="Number of scientist-loop cycles when enabled.")
    scientist_discovery_mode: str | None = Field(
        default=None,
        description="Optional scientist discovery mode: standard, hamiltonian, lagrangian, or structured_lagrangian.",
    )
    scientist_discover_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional kwargs forwarded to scientist discovery backend.",
    )
    scientist_use_active_inference: bool = Field(
        default=False,
        description="Enable active-inference model scoring and experiment selection in scientist loop.",
    )
    scientist_active_inference_strategy: str = Field(
        default="uncertainty_sampling",
        description="Active inference strategy (e.g., uncertainty_sampling, information_gain, random).",
    )
    scientist_available_experiments: list[str] | None = Field(
        default=None,
        description="Optional pool of dataset paths for active-inference experiment selection.",
    )


class AskOutput(BaseModel):
    result: dict[str, Any] = Field(default_factory=dict)
