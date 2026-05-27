from __future__ import annotations

from dataclasses import asdict

from physics_engine.discovery import discover_law
from physics_engine.knowledge import load_knowledge_base, search_laws
from physics_engine.reasoning import (
    generate_hypotheses,
    generate_hypothesis_explanation,
    validate_candidate_law,
)
from physics_engine.scientist import AIScientist


def _looks_like_discovery_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "derive",
        "discover",
        "from data",
        "fit law",
        "equation from",
        "dataset",
    )
    return any(key in text for key in keys)


def _looks_like_scientist_loop_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "scientist loop",
        "automated scientist",
        "run cycle",
        "closed-loop",
    )
    return any(key in text for key in keys)


def _looks_like_hamiltonian_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "hamiltonian",
        "canonical",
        "phase space",
        "symplectic",
    )
    return any(key in text for key in keys)


def _looks_like_lagrangian_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "lagrangian",
        "euler-lagrange",
        "action principle",
        "l = t - v",
    )
    return any(key in text for key in keys)


def _looks_like_structured_lagrangian_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "structured lagrangian",
        "t(dq)",
        "t-v split",
        "kinetic and potential",
        "structured variational",
    )
    return any(key in text for key in keys)


def _looks_like_multibody_lagrangian_request(question: str) -> bool:
    text = (question or "").lower()
    keys = (
        "multibody",
        "multi-body",
        "robot arm",
        "2-link",
        "2 dof",
        "coupled oscillator",
        "inertia matrix",
        "mass matrix",
    )
    return any(key in text for key in keys)


def answer_physics_question(
    question: str,
    dataset_path: str | None = None,
    target_var: str | None = None,
    domain: str | None = None,
    run_discovery: bool = True,
    run_scientist_loop: bool = True,
    scientist_cycles: int = 3,
    scientist_discovery_mode: str | None = None,
    scientist_discover_kwargs: dict | None = None,
    scientist_use_active_inference: bool = False,
    scientist_active_inference_strategy: str = "uncertainty_sampling",
    scientist_available_experiments: list[str] | None = None,
) -> dict:
    known_matches = search_laws(question, domain=domain, top_k=5)
    hypotheses = generate_hypotheses(question=question, known_laws=known_matches)

    wants_discovery = run_discovery and _looks_like_discovery_request(question)
    wants_scientist_loop = run_scientist_loop and _looks_like_scientist_loop_request(question)
    wants_hamiltonian = _looks_like_hamiltonian_request(question)
    wants_lagrangian = _looks_like_lagrangian_request(question)
    wants_structured_lagrangian = _looks_like_structured_lagrangian_request(question)
    wants_multibody_lagrangian = _looks_like_multibody_lagrangian_request(question)
    can_discover = bool(dataset_path and target_var)

    selected_scientist_mode = (scientist_discovery_mode or "").strip().lower()
    if selected_scientist_mode not in {
        "standard",
        "hamiltonian",
        "lagrangian",
        "structured_lagrangian",
        "multibody_lagrangian",
    }:
        if wants_structured_lagrangian:
            selected_scientist_mode = "structured_lagrangian"
        elif wants_multibody_lagrangian:
            selected_scientist_mode = "multibody_lagrangian"
        elif wants_lagrangian:
            selected_scientist_mode = "lagrangian"
        elif wants_hamiltonian:
            selected_scientist_mode = "hamiltonian"
        else:
            selected_scientist_mode = "standard"

    discover_kwargs = scientist_discover_kwargs or {}

    response = {
        "mode": "knowledge",
        "question": question,
        "answer": "",
        "known_law_matches": [entry.to_dict() for entry in known_matches],
        "hypotheses": [asdict(item) for item in hypotheses],
        "structured_decomposition": None,
        "active_inference": None,
        "latest_metadata_summary": None,
    }

    if wants_scientist_loop and can_discover:
        scientist = AIScientist(confidence_threshold=0.9)
        run_kwargs = {
            "dataset_path": dataset_path,
            "target_var": target_var,
            "cycles": max(1, int(scientist_cycles)),
            "question": question,
            "domain": domain,
            "discovery_mode": selected_scientist_mode,
            "discover_kwargs": discover_kwargs,
        }
        if scientist_use_active_inference or scientist_available_experiments is not None:
            run_kwargs["use_active_inference"] = bool(scientist_use_active_inference)
            run_kwargs["active_inference_strategy"] = str(scientist_active_inference_strategy)
            run_kwargs["available_experiments"] = scientist_available_experiments

        try:
            loop_result = scientist.run(**run_kwargs)
        except TypeError:
            run_kwargs.pop("use_active_inference", None)
            run_kwargs.pop("active_inference_strategy", None)
            run_kwargs.pop("available_experiments", None)
            loop_result = scientist.run(**run_kwargs)
        last_cycle = loop_result["cycles"][-1] if loop_result["cycles"] else {}
        symmetry_summary = ""
        symmetry = last_cycle.get("symmetry_validation")
        if isinstance(symmetry, dict):
            rotational = bool(symmetry.get("rotational_symmetry", False))
            conserved = bool(symmetry.get("angular_momentum_conserved", False))
            sym_conf = float(symmetry.get("confidence", 0.0))
            symmetry_summary = (
                f" Symmetry: rotational={rotational}, angular_momentum_conserved={conserved}, "
                f"symmetry_confidence={sym_conf:.3f}."
            )
        conservation_summary = ""
        conservation = last_cycle.get("conservation_validation")
        if isinstance(conservation, dict):
            detected = bool(conservation.get("detected", False))
            quantity = str(conservation.get("quantity_name", "n/a"))
            score = float(conservation.get("score", 0.0))
            conservation_summary = (
                f" Conservation: detected={detected}, quantity={quantity}, score={score:.6f}."
            )
        decomposition_summary = ""
        decomposition = last_cycle.get("structured_decomposition")
        if isinstance(decomposition, dict):
            kinetic = str(decomposition.get("kinetic_energy", "")).strip()
            potential = str(decomposition.get("potential_energy", "")).strip()
            mass_matrix = decomposition.get("mass_matrix")
            response_decomposition: dict[str, object] = {}
            if kinetic:
                response_decomposition["kinetic_energy"] = kinetic
            if potential:
                response_decomposition["potential_energy"] = potential
            if isinstance(mass_matrix, dict):
                response_decomposition["mass_matrix"] = {
                    "M11": str(mass_matrix.get("M11", "")).strip(),
                    "M12": str(mass_matrix.get("M12", "")).strip(),
                    "M22": str(mass_matrix.get("M22", "")).strip(),
                }

            if response_decomposition:
                response["structured_decomposition"] = response_decomposition

            if kinetic or potential:
                decomposition_summary = (
                    f" Structured decomposition: T(dq)={kinetic or 'n/a'}; V(q)={potential or 'n/a'}."
                )
            elif isinstance(mass_matrix, dict):
                decomposition_summary = (
                    " Structured decomposition: "
                    f"M11={str(mass_matrix.get('M11', 'n/a')).strip() or 'n/a'}, "
                    f"M12={str(mass_matrix.get('M12', 'n/a')).strip() or 'n/a'}, "
                    f"M22={str(mass_matrix.get('M22', 'n/a')).strip() or 'n/a'}."
                )
        response["mode"] = "scientist_loop"
        response["scientist_loop"] = loop_result
        response["scientist_discovery_mode"] = selected_scientist_mode

        active_inference_payload = loop_result.get("active_inference")
        if isinstance(active_inference_payload, dict):
            response["active_inference"] = active_inference_payload

        latest_metadata_summary = None
        metadata_payload = loop_result.get("metadata_records")
        if isinstance(metadata_payload, dict):
            records = metadata_payload.get("records")
            if isinstance(records, list) and records:
                latest = records[-1]
                if isinstance(latest, dict):
                    discovered = latest.get("discovered_law")
                    metrics = latest.get("metrics")
                    latest_metadata_summary = {
                        "experiment_name": latest.get("experiment_name"),
                        "timestamp": latest.get("timestamp"),
                        "equation": (
                            discovered.get("equation")
                            if isinstance(discovered, dict)
                            else None
                        ),
                        "training_mse": (
                            metrics.get("training_mse")
                            if isinstance(metrics, dict)
                            else None
                        ),
                        "free_energy": (
                            metrics.get("free_energy")
                            if isinstance(metrics, dict)
                            else None
                        ),
                        "confidence": (
                            metrics.get("confidence")
                            if isinstance(metrics, dict)
                            else None
                        ),
                    }
                    response["latest_metadata_summary"] = latest_metadata_summary

        active_summary = ""
        if isinstance(active_inference_payload, dict):
            enabled = bool(active_inference_payload.get("enabled", False))
            if enabled:
                strategy = str(active_inference_payload.get("strategy", "n/a"))
                model_count = int(active_inference_payload.get("model_count", 0))
                uncertainty = float(active_inference_payload.get("epistemic_uncertainty", 0.0))
                active_summary = (
                    f" Active inference: strategy={strategy}, models={model_count}, "
                    f"epistemic_uncertainty={uncertainty:.6f}."
                )

        metadata_summary_text = ""
        if isinstance(latest_metadata_summary, dict):
            metadata_summary_text = (
                " Metadata: "
                f"experiment={latest_metadata_summary.get('experiment_name', 'n/a')}, "
                f"mse={float(latest_metadata_summary.get('training_mse') or 0.0):.6f}, "
                f"free_energy={float(latest_metadata_summary.get('free_energy') or 0.0):.6f}."
            )

        response["answer"] = (
            f"Scientist loop completed {len(loop_result['cycles'])} cycle(s). "
            f"Latest equation: {last_cycle.get('discovered_equation', 'n/a')}. "
            f"Latest confidence: {last_cycle.get('confidence', 0.0):.3f}."
            f"{symmetry_summary}{conservation_summary}{decomposition_summary}{active_summary}{metadata_summary_text}"
        )
        return response

    if wants_discovery and can_discover:
        law, error, significant = discover_law(dataset_path, target_var)
        known_base = load_knowledge_base(domains=(domain,) if domain else None)
        discovery_hypotheses = generate_hypotheses(
            question=question,
            known_laws=known_matches,
            discovered_equation=str(law),
        )
        validation = validate_candidate_law(
            discovered_equation=str(law),
            validation_error=float(error),
            known_laws=known_base,
            significant_correlations=significant,
        )
        explanation = generate_hypothesis_explanation(discovery_hypotheses, validation)
        response["mode"] = "discovery"
        response["discovered_equation"] = str(law)
        response["validation_error"] = float(error)
        response["significant_residual_correlations"] = {
            str(key): float(value) for key, value in significant.items()
        }
        response["hypotheses"] = [asdict(item) for item in discovery_hypotheses]
        response["validation"] = asdict(validation)
        response["reasoning_explanation"] = explanation
        response["answer"] = (
            f"Discovered equation for {target_var}: {law}. "
            f"Validation score={validation.overall_score:.3f}."
        )
        return response

    if known_matches:
        top = known_matches[0]
        response["answer"] = f"Closest known law: {top.law} ({top.equation})."
    else:
        response["answer"] = (
            "No close law found in the knowledge base. Provide dataset_path and target_var "
            "to run discovery."
        )

    return response
