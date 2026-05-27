from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from physics_engine.discovery import (
    discover_hamiltonian_from_dataframe,
    discover_lagrangian_from_dataframe,
    discover_law,
)
from physics_engine.knowledge_graph import PhysicsKnowledgeGraph, RelationType
from physics_engine.knowledge import load_knowledge_base
from physics_engine.active_inference import (
    BeliefState,
    ExperimentMetadataLogger,
    ExperimentSelector,
    compute_free_energy,
)
from physics_engine.reasoning import (
    detect_conserved_quantities,
    validate_candidate_law,
    validate_orbit_symmetry,
)
from physics_engine.variational import discover_structured_lagrangian_from_dataframe
from physics_engine.variational import discover_multibody_lagrangian_from_dataframe

from .experiment_designer import ExperimentProposal, design_next_experiment
from .hypothesis_generator import generate_guided_hypotheses
from .theory_manager import Theory, TheoryManager


@dataclass(frozen=True)
class ScientistCycleResult:
    cycle: int
    dataset_path: str
    target_var: str
    discovery_mode: str
    discovered_equation: str
    validation_error: float
    discovery_metrics: dict[str, float]
    confidence: float
    closest_known_law: str | None
    stored_as_theory: bool
    hypotheses: tuple[str, ...]
    next_experiment: ExperimentProposal
    symmetry_validation: dict[str, float | bool] | None = None
    conservation_validation: dict[str, float | bool | str] | None = None
    structured_decomposition: dict[str, object] | None = None
    active_inference_free_energy: float | None = None
    active_inference_selected_experiment: str | None = None
    active_inference_reasoning: str | None = None
    metadata_record_id: str | None = None


class AIScientist:
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        theory_manager: TheoryManager | None = None,
        knowledge_graph: PhysicsKnowledgeGraph | None = None,
        discovery_fn: Callable[..., tuple[object, float, dict[str, float]]] | None = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.theory_manager = theory_manager or TheoryManager()
        self.knowledge_graph = knowledge_graph or PhysicsKnowledgeGraph()
        self.discovery_fn = discovery_fn or discover_law
        self.cycle_log: list[ScientistCycleResult] = []
        self.metadata_logger = ExperimentMetadataLogger()

    def _estimate_primary_mse(self, discovery_metrics: dict[str, float], fallback_error: float) -> float:
        for key in (
            "tau_mse",
            "residual_mse",
            "euler_lagrange_mse",
            "dqdt_mse",
            "dpdt_mse",
        ):
            value = discovery_metrics.get(key)
            if value is not None:
                return max(float(value), 0.0)
        return max(float(fallback_error), 0.0)

    def _estimate_parameter_count(
        self,
        equation: str,
        structured_decomposition: dict[str, object] | None,
    ) -> int:
        base = len(self._extract_law_variables(equation))
        if structured_decomposition:
            mass_matrix = structured_decomposition.get("mass_matrix")
            if isinstance(mass_matrix, dict):
                base += len([k for k, v in mass_matrix.items() if str(v).strip()])
            if str(structured_decomposition.get("kinetic_energy", "")).strip():
                base += 1
            if str(structured_decomposition.get("potential_energy", "")).strip():
                base += 1
        return max(1, int(base))

    def _dataset_metadata(
        self,
        dataset_path: str,
        target_var: str,
        domain: str | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "path": str(dataset_path),
            "domain": domain or "unknown",
        }
        try:
            df = pd.read_csv(dataset_path)
            metadata["num_samples"] = int(len(df))
            metadata["num_features"] = int(max(0, len(df.columns) - 1))
            metadata["num_outputs"] = 1
            metadata["notes"] = f"target_var={target_var}"
        except Exception:
            metadata["notes"] = f"target_var={target_var}; dataset_read_failed"
        return metadata

    def _slugify(self, value: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
        return text.strip("_") or "node"

    def _extract_law_variables(self, equation: str) -> list[str]:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", equation))
        blocked = {
            "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "e",
            "true", "false",
        }
        return sorted(token for token in tokens if token.lower() not in blocked)

    def _run_hamiltonian_discovery(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> tuple[str, float, dict[str, float]]:
        q_col = str(discover_kwargs.get("q_col", "q"))
        p_col = str(discover_kwargs.get("p_col", "p"))
        dqdt_col = str(discover_kwargs.get("dqdt_col", "dqdt"))
        dpdt_col = str(discover_kwargs.get("dpdt_col", "dpdt"))

        h_kwargs = {
            "max_power": int(discover_kwargs.get("max_power", 2)),
            "include_cross_terms": bool(discover_kwargs.get("include_cross_terms", True)),
        }

        df = pd.read_csv(dataset_path)
        result = discover_hamiltonian_from_dataframe(
            df=df,
            q_col=q_col,
            p_col=p_col,
            dqdt_col=dqdt_col,
            dpdt_col=dpdt_col,
            **h_kwargs,
        )
        metrics = {
            "dqdt_mse": float(result.dqdt_mse),
            "dpdt_mse": float(result.dpdt_mse),
        }
        return result.hamiltonian, float(result.error), metrics

    def _run_lagrangian_discovery(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> tuple[str, float, dict[str, float]]:
        q_col = str(discover_kwargs.get("q_col", "q"))
        dqdt_col = str(discover_kwargs.get("dqdt_col", "dqdt"))
        d2qdt2_col = str(discover_kwargs.get("d2qdt2_col", "d2qdt2"))

        l_kwargs = {
            "max_power": int(discover_kwargs.get("max_power", 2)),
            "include_cross_terms": bool(discover_kwargs.get("include_cross_terms", True)),
        }

        df = pd.read_csv(dataset_path)
        result = discover_lagrangian_from_dataframe(
            df=df,
            q_col=q_col,
            dqdt_col=dqdt_col,
            d2qdt2_col=d2qdt2_col,
            **l_kwargs,
        )
        metrics = {
            "euler_lagrange_mse": float(result.euler_lagrange_mse),
        }
        return result.lagrangian, float(result.error), metrics

    def _run_structured_lagrangian_discovery(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> tuple[str, float, dict[str, float], dict[str, str]]:
        q_col = str(discover_kwargs.get("q_col", "q"))
        dqdt_col = str(discover_kwargs.get("dqdt_col", "dqdt"))
        d2qdt2_col = str(discover_kwargs.get("d2qdt2_col", "d2qdt2"))

        sl_kwargs = {
            "kinetic_max_even_power": int(discover_kwargs.get("kinetic_max_even_power", 4)),
            "potential_max_even_power": int(discover_kwargs.get("potential_max_even_power", 4)),
            "include_trig": bool(discover_kwargs.get("include_trig", True)),
        }

        df = pd.read_csv(dataset_path)
        result = discover_structured_lagrangian_from_dataframe(
            df=df,
            q_col=q_col,
            dq_col=dqdt_col,
            ddq_col=d2qdt2_col,
            **sl_kwargs,
        )
        metrics = {
            "residual_mse": float(result.residual_mse),
            "residual_rmse": float(result.residual_rmse),
        }
        decomposition = {
            "kinetic_energy": str(result.kinetic_energy),
            "potential_energy": str(result.potential_energy),
        }
        return result.lagrangian, float(result.residual_mse), metrics, decomposition

    def _run_multibody_lagrangian_discovery(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> tuple[str, float, dict[str, float], dict[str, object]]:
        q1_col = str(discover_kwargs.get("q1_col", "q1"))
        q2_col = str(discover_kwargs.get("q2_col", "q2"))
        dq1_col = str(discover_kwargs.get("dq1_col", "dq1"))
        dq2_col = str(discover_kwargs.get("dq2_col", "dq2"))
        ddq1_col = str(discover_kwargs.get("ddq1_col", "ddq1"))
        ddq2_col = str(discover_kwargs.get("ddq2_col", "ddq2"))
        tau1_col = str(discover_kwargs.get("tau1_col", "tau1"))
        tau2_col = str(discover_kwargs.get("tau2_col", "tau2"))

        df = pd.read_csv(dataset_path)
        result = discover_multibody_lagrangian_from_dataframe(
            df=df,
            q1_col=q1_col,
            q2_col=q2_col,
            dq1_col=dq1_col,
            dq2_col=dq2_col,
            ddq1_col=ddq1_col,
            ddq2_col=ddq2_col,
            tau1_col=tau1_col,
            tau2_col=tau2_col,
        )

        metrics = {
            "tau_mse": float(result.tau_mse),
            "tau_rmse": float(result.tau_rmse),
            "mass_spd_fraction": float(result.mass_spd_fraction),
        }

        m11 = str(result.mass_matrix.get("M11", "0"))
        m12 = str(result.mass_matrix.get("M12", "0"))
        m22 = str(result.mass_matrix.get("M22", "0"))
        kinetic_expr = f"0.5*({m11})*dq1**2 + ({m12})*dq1*dq2 + 0.5*({m22})*dq2**2"

        decomposition: dict[str, object] = {
            "mass_matrix": {
                "M11": m11,
                "M12": m12,
                "M22": m22,
            },
            "kinetic_energy": kinetic_expr,
            "potential_energy": str(result.potential_energy),
        }
        return result.lagrangian, float(result.tau_mse), metrics, decomposition

    def _log_to_knowledge_graph(
        self,
        *,
        dataset_path: str,
        target_var: str,
        domain: str | None,
        discovery_mode: str,
        equation: str,
        validation_error: float,
        confidence: float,
        cycle_index: int,
        stored_as_theory: bool,
        closest_known_law: str | None,
        discover_kwargs: dict,
        conservation_validation: dict[str, float | bool | str] | None,
        structured_decomposition: dict[str, object] | None,
    ) -> None:
        dataset_slug = self._slugify(Path(dataset_path).stem)
        law_slug = self._slugify(f"{target_var}_{discovery_mode}_{cycle_index}")
        experiment_slug = self._slugify(f"exp_{target_var}_{discovery_mode}_{cycle_index}")
        algorithm_slug = self._slugify(discovery_mode)

        dataset_size = discover_kwargs.get("dataset_size", 0)
        noise_level = discover_kwargs.get("noise_level", 0.0)

        self.knowledge_graph.add_dataset(
            dataset_slug,
            path=str(dataset_path),
            samples=int(dataset_size) if dataset_size else 0,
            noise_level=float(noise_level) if noise_level is not None else 0.0,
            domain=domain or "unknown",
            metadata={"target_var": target_var},
        )
        self.knowledge_graph.add_algorithm(
            algorithm_slug,
            display_name=discovery_mode,
            family="variational" if "lagrangian" in discovery_mode else discovery_mode,
            description=f"Scientist-loop discovery mode: {discovery_mode}",
        )
        self.knowledge_graph.add_experiment(
            experiment_slug,
            experiment_id=experiment_slug,
            algorithm=discovery_mode,
            dataset=str(dataset_path),
            success=stored_as_theory,
            metadata={
                "target_var": target_var,
                "cycle": cycle_index,
                "confidence": confidence,
                "validation_error": validation_error,
            },
        )
        self.knowledge_graph.add_law(
            law_slug,
            equation=equation,
            variables=self._extract_law_variables(equation),
            domain=domain or "unknown",
            source="scientist_loop",
            confidence=confidence,
            error=validation_error,
            metadata={
                "target_var": target_var,
                "discovery_mode": discovery_mode,
                "cycle": cycle_index,
                "stored_as_theory": stored_as_theory,
            },
        )

        self.knowledge_graph.add_relation(law_slug, RelationType.VALIDATED_ON, dataset_slug)
        self.knowledge_graph.add_relation(law_slug, RelationType.DISCOVERED_BY, algorithm_slug)
        self.knowledge_graph.add_relation(law_slug, RelationType.DISCOVERED_FROM, experiment_slug)

        if structured_decomposition:
            kinetic_expr = str(structured_decomposition.get("kinetic_energy", "")).strip()
            potential_expr = str(structured_decomposition.get("potential_energy", "")).strip()

            if kinetic_expr:
                kinetic_slug = self._slugify(f"{target_var}_{discovery_mode}_{cycle_index}_kinetic")
                self.knowledge_graph.add_law(
                    kinetic_slug,
                    equation=kinetic_expr,
                    variables=self._extract_law_variables(kinetic_expr),
                    domain=domain or "unknown",
                    source="scientist_loop_decomposition",
                    confidence=confidence,
                    error=validation_error,
                    metadata={
                        "target_var": target_var,
                        "discovery_mode": discovery_mode,
                        "cycle": cycle_index,
                        "component_type": "kinetic_energy",
                    },
                )
                self.knowledge_graph.add_relation(
                    law_slug,
                    RelationType.DERIVED_FROM,
                    kinetic_slug,
                    metadata={"component": "T"},
                )

            if potential_expr:
                potential_slug = self._slugify(f"{target_var}_{discovery_mode}_{cycle_index}_potential")
                self.knowledge_graph.add_law(
                    potential_slug,
                    equation=potential_expr,
                    variables=self._extract_law_variables(potential_expr),
                    domain=domain or "unknown",
                    source="scientist_loop_decomposition",
                    confidence=confidence,
                    error=validation_error,
                    metadata={
                        "target_var": target_var,
                        "discovery_mode": discovery_mode,
                        "cycle": cycle_index,
                        "component_type": "potential_energy",
                    },
                )
                self.knowledge_graph.add_relation(
                    law_slug,
                    RelationType.DERIVED_FROM,
                    potential_slug,
                    metadata={"component": "V"},
                )

            mass_matrix = structured_decomposition.get("mass_matrix")
            if isinstance(mass_matrix, dict):
                for entry_name in ("M11", "M12", "M22"):
                    entry_expr = str(mass_matrix.get(entry_name, "")).strip()
                    if not entry_expr:
                        continue
                    entry_slug = self._slugify(f"{target_var}_{discovery_mode}_{cycle_index}_{entry_name}")
                    self.knowledge_graph.add_law(
                        entry_slug,
                        equation=entry_expr,
                        variables=self._extract_law_variables(entry_expr),
                        domain=domain or "unknown",
                        source="scientist_loop_decomposition",
                        confidence=confidence,
                        error=validation_error,
                        metadata={
                            "target_var": target_var,
                            "discovery_mode": discovery_mode,
                            "cycle": cycle_index,
                            "component_type": "mass_matrix_entry",
                            "mass_entry": entry_name,
                        },
                    )
                    self.knowledge_graph.add_relation(
                        law_slug,
                        RelationType.DERIVED_FROM,
                        entry_slug,
                        metadata={"component": entry_name},
                    )

        if closest_known_law:
            known_slug = self._slugify(closest_known_law)
            if self.knowledge_graph.get_node(known_slug) is None:
                self.knowledge_graph.add_law(
                    known_slug,
                    equation=closest_known_law,
                    domain=domain or "unknown",
                    source="known",
                    confidence=1.0,
                    metadata={"display_name": closest_known_law},
                )
            self.knowledge_graph.add_relation(law_slug, RelationType.CONSISTENT_WITH, known_slug)

        if conservation_validation and bool(conservation_validation.get("detected", False)):
            quantity_name = str(conservation_validation.get("quantity_name", "conserved_quantity"))
            expression = str(conservation_validation.get("expression", ""))
            invariant_slug = self._slugify(f"{quantity_name}_{target_var}_{cycle_index}")
            self.knowledge_graph.add_invariant(
                invariant_slug,
                expression=expression,
                conservation_law=quantity_name,
                metadata={
                    "derived_from_mode": discovery_mode,
                    "score": float(conservation_validation.get("score", 0.0)),
                    "threshold": float(conservation_validation.get("threshold", 0.0)),
                },
            )
            self.knowledge_graph.add_relation(law_slug, RelationType.CONSERVES, invariant_slug)

    def _compute_symmetry_validation(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> dict[str, float | bool] | None:
        x_col = str(discover_kwargs.get("x_col", "x"))
        y_col = str(discover_kwargs.get("y_col", "y"))
        vx_col = str(discover_kwargs.get("vx_col", "vx"))
        vy_col = str(discover_kwargs.get("vy_col", "vy"))
        ax_col = str(discover_kwargs.get("ax_col", "ax"))
        ay_col = str(discover_kwargs.get("ay_col", "ay"))

        try:
            df = pd.read_csv(dataset_path)
        except Exception:
            return None

        required_cols = (x_col, y_col, vx_col, vy_col, ax_col, ay_col)
        if not all(col in df.columns for col in required_cols):
            return None

        orbit_df = df[[x_col, y_col, vx_col, vy_col, ax_col, ay_col]].rename(
            columns={
                x_col: "x",
                y_col: "y",
                vx_col: "vx",
                vy_col: "vy",
                ax_col: "ax",
                ay_col: "ay",
            }
        )

        dt = float(discover_kwargs.get("dt", 1.0))
        mass = float(discover_kwargs.get("mass", 1.0))
        symmetry_tolerance = float(discover_kwargs.get("symmetry_tolerance", 0.15))
        conservation_tolerance = float(discover_kwargs.get("conservation_tolerance", 1e-3))

        try:
            result = validate_orbit_symmetry(
                orbit_df,
                dt=dt,
                mass=mass,
                symmetry_tolerance=symmetry_tolerance,
                conservation_tolerance=conservation_tolerance,
            )
            return result.to_dict()
        except Exception:
            return {
                "rotational_symmetry": False,
                "rotational_symmetry_score": float("inf"),
                "rotational_symmetry_coverage": 0.0,
                "angular_momentum_conserved": False,
                "angular_momentum_score": float("inf"),
                "confidence": 0.0,
            }

    def _compute_conservation_validation(
        self,
        dataset_path: str,
        discover_kwargs: dict,
    ) -> dict[str, float | bool | str] | None:
        try:
            df = pd.read_csv(dataset_path)
        except Exception:
            return None

        threshold = float(discover_kwargs.get("conservation_tolerance", 1e-3))
        time_col = discover_kwargs.get("time_col")
        time_col = str(time_col) if time_col is not None else None

        try:
            result = detect_conserved_quantities(
                df,
                threshold=threshold,
                time_col=time_col,
            )
        except Exception:
            return None

        return result.to_dict() if result is not None else None

    def run_cycle(
        self,
        dataset_path: str,
        target_var: str,
        question: str = "",
        domain: str | None = None,
        cycle_index: int = 1,
        discovery_mode: str = "standard",
        discover_kwargs: dict | None = None,
        use_active_inference: bool = False,
        belief_state: BeliefState | None = None,
        active_inference_selection_reason: str | None = None,
    ) -> ScientistCycleResult:
        cycle_start = time.perf_counter()
        discover_kwargs = discover_kwargs or {}
        structured_decomposition: dict[str, object] | None = None

        hypotheses = generate_guided_hypotheses(
            target_var=target_var,
            question=question,
            domain=domain,
            max_candidates=5,
        )

        mode = str(discovery_mode).strip().lower() or "standard"
        if mode == "hamiltonian":
            law, error, discovery_metrics = self._run_hamiltonian_discovery(
                dataset_path=dataset_path,
                discover_kwargs=discover_kwargs,
            )
            significant: dict[str, float] = {}
        elif mode == "structured_lagrangian":
            law, error, discovery_metrics, structured_decomposition = self._run_structured_lagrangian_discovery(
                dataset_path=dataset_path,
                discover_kwargs=discover_kwargs,
            )
            significant = {}
        elif mode == "multibody_lagrangian":
            law, error, discovery_metrics, structured_decomposition = self._run_multibody_lagrangian_discovery(
                dataset_path=dataset_path,
                discover_kwargs=discover_kwargs,
            )
            significant = {}
        elif mode == "lagrangian":
            law, error, discovery_metrics = self._run_lagrangian_discovery(
                dataset_path=dataset_path,
                discover_kwargs=discover_kwargs,
            )
            significant = {}
        else:
            law, error, significant = self.discovery_fn(dataset_path, target_var, **discover_kwargs)
            discovery_metrics = {
                "num_significant_correlations": float(len(significant)),
            }

        known_laws = load_knowledge_base(domains=(domain,) if domain else None)
        validation = validate_candidate_law(
            discovered_equation=str(law),
            validation_error=float(error),
            known_laws=known_laws,
            significant_correlations=significant,
            dimensionally_plausible=True,
        )

        should_store = validation.overall_score >= self.confidence_threshold
        if should_store:
            theory = Theory(
                name=validation.closest_known_law or f"candidate_{target_var}",
                equation=str(law),
                confidence=float(validation.overall_score),
                supporting_datasets=(str(dataset_path),),
                domain=domain,
            )
            self.theory_manager.add_theory(theory)

        conservation_validation = self._compute_conservation_validation(
            dataset_path=dataset_path,
            discover_kwargs=discover_kwargs,
        )

        self._log_to_knowledge_graph(
            dataset_path=dataset_path,
            target_var=target_var,
            domain=domain,
            discovery_mode=mode,
            equation=str(law),
            validation_error=float(error),
            confidence=float(validation.overall_score),
            cycle_index=cycle_index,
            stored_as_theory=bool(should_store),
            closest_known_law=validation.closest_known_law,
            discover_kwargs=discover_kwargs,
            conservation_validation=conservation_validation,
            structured_decomposition=structured_decomposition,
        )

        next_experiment = design_next_experiment(
            cycle_index=cycle_index,
            confidence=float(validation.overall_score),
            validation_error=float(error),
        )

        active_inference_free_energy: float | None = None
        if use_active_inference:
            mse_value = self._estimate_primary_mse(discovery_metrics, fallback_error=float(error))
            n_params = self._estimate_parameter_count(str(law), structured_decomposition)
            synthetic_observed = np.zeros(16, dtype=float)
            synthetic_predicted = np.full(16, np.sqrt(mse_value), dtype=float)
            fe = compute_free_energy(
                observed=synthetic_observed,
                predicted=synthetic_predicted,
                sigma=float(discover_kwargs.get("sigma", 1.0)),
                n_params=n_params,
                regularization=float(discover_kwargs.get("free_energy_regularization", 1.0)),
            )
            active_inference_free_energy = float(fe.f_value)
            if belief_state is not None:
                belief_state.add_model(
                    model_id=f"cycle_{cycle_index}",
                    law_dict={
                        "equation": str(law),
                        "variables": self._extract_law_variables(str(law)),
                        "coefficients": {},
                        "domain": domain or "unknown",
                    },
                    free_energy=active_inference_free_energy,
                    n_params=n_params,
                    mse=float(mse_value),
                    metadata={
                        "dataset_path": str(dataset_path),
                        "discovery_mode": mode,
                    },
                )

        runtime_seconds = float(time.perf_counter() - cycle_start)
        experiment_name = f"cycle_{cycle_index}_{self._slugify(target_var)}_{mode}"
        metadata_record = self.metadata_logger.create_experiment_record(
            experiment_name=experiment_name,
            dataset_name=Path(dataset_path).stem,
            discovery_method=mode,
            discovered_law={
                "equation": str(law),
                "variables": self._extract_law_variables(str(law)),
                "coefficients": {
                    key: float(value)
                    for key, value in discovery_metrics.items()
                    if isinstance(value, (int, float))
                },
            },
            metrics={
                **discovery_metrics,
                "mse": self._estimate_primary_mse(discovery_metrics, fallback_error=float(error)),
                "validation_mse": float(error),
                "confidence": float(validation.overall_score),
                "free_energy": float(active_inference_free_energy or 0.0),
            },
            dataset_metadata=self._dataset_metadata(
                dataset_path=dataset_path,
                target_var=target_var,
                domain=domain,
            ),
            algorithm_metadata={
                "name": mode,
                "discovery_mode": mode,
                "hyperparameters": dict(discover_kwargs),
                "notes": f"cycle={cycle_index}",
            },
            runtime_seconds=runtime_seconds,
            notes=question,
            seed=discover_kwargs.get("seed"),
        )

        symmetry_validation = self._compute_symmetry_validation(
            dataset_path=dataset_path,
            discover_kwargs=discover_kwargs,
        )
        result = ScientistCycleResult(
            cycle=cycle_index,
            dataset_path=str(dataset_path),
            target_var=target_var,
            discovery_mode=mode,
            discovered_equation=str(law),
            validation_error=float(error),
            discovery_metrics=discovery_metrics,
            confidence=float(validation.overall_score),
            closest_known_law=validation.closest_known_law,
            stored_as_theory=bool(should_store),
            hypotheses=tuple(hypotheses),
            next_experiment=next_experiment,
            symmetry_validation=symmetry_validation,
            conservation_validation=conservation_validation,
            structured_decomposition=structured_decomposition,
            active_inference_free_energy=active_inference_free_energy,
            active_inference_selected_experiment=str(dataset_path) if use_active_inference else None,
            active_inference_reasoning=active_inference_selection_reason,
            metadata_record_id=metadata_record.experiment_name,
        )
        self.cycle_log.append(result)
        return result

    def run(
        self,
        dataset_path: str,
        target_var: str,
        cycles: int = 3,
        question: str = "",
        domain: str | None = None,
        discovery_mode: str = "standard",
        discover_kwargs: dict | None = None,
        use_active_inference: bool = False,
        active_inference_strategy: str = "uncertainty_sampling",
        available_experiments: list[str] | None = None,
    ) -> dict:
        discover_kwargs = discover_kwargs or {}
        results: list[ScientistCycleResult] = []
        belief_state: BeliefState | None = BeliefState() if use_active_inference else None
        selector: ExperimentSelector | None = (
            ExperimentSelector(strategy=active_inference_strategy)
            if use_active_inference
            else None
        )

        experiment_pool = available_experiments or [str(dataset_path)]

        for cycle_idx in range(1, max(1, int(cycles)) + 1):
            chosen_dataset = str(dataset_path)
            selection_reason = None
            if use_active_inference and selector is not None and belief_state is not None:
                chosen_dataset, score = selector.choose_experiment(
                    belief_state=belief_state,
                    available_experiments=experiment_pool,
                )
                selection_reason = score.reasoning

            cycle_result = self.run_cycle(
                dataset_path=chosen_dataset,
                target_var=target_var,
                question=question,
                domain=domain,
                cycle_index=cycle_idx,
                discovery_mode=discovery_mode,
                discover_kwargs=discover_kwargs,
                use_active_inference=use_active_inference,
                belief_state=belief_state,
                active_inference_selection_reason=selection_reason,
            )
            results.append(cycle_result)

        return {
            "cycles": [
                {
                    **asdict(item),
                    "next_experiment": asdict(item.next_experiment),
                }
                for item in results
            ],
            "theories": self.theory_manager.to_dict(),
            "knowledge_graph": self.knowledge_graph.to_dict(),
            "knowledge_graph_stats": self.knowledge_graph.stats(),
            "metadata_records": self.metadata_logger.to_dict(),
            "active_inference": {
                "enabled": bool(use_active_inference),
                "strategy": active_inference_strategy if use_active_inference else None,
                "model_count": belief_state.model_count() if belief_state is not None else 0,
                "best_model_id": (
                    belief_state.best_model().id
                    if (belief_state is not None and belief_state.best_model() is not None)
                    else None
                ),
                "epistemic_uncertainty": (
                    belief_state.epistemic_uncertainty()
                    if belief_state is not None
                    else 0.0
                ),
            },
        }

    def export_log(self, output_path: str | Path) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cycles": [
                {
                    **asdict(item),
                    "next_experiment": asdict(item.next_experiment),
                }
                for item in self.cycle_log
            ],
            "theories": self.theory_manager.to_dict(),
            "knowledge_graph": self.knowledge_graph.to_dict(),
            "knowledge_graph_stats": self.knowledge_graph.stats(),
            "metadata_records": self.metadata_logger.to_dict(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def export_knowledge_graph(self, output_path: str | Path) -> Path:
        """Persist the current scientist knowledge graph to JSON."""
        return self.knowledge_graph.save(output_path)
