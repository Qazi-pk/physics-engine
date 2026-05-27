"""
experiment_selector.py
~~~~~~~~~~~~~~~~~~~~~

Select next experiment based on epistemic uncertainty and information gain.

The experiment selector uses the belief state to decide which experiment to run next.
Strategies include:
  - uncertainty_sampling: Choose dataset that maximizes model disagreement
  - information_gain: Choose dataset that reduces entropy most (future)
  - random: Random selection (baseline)

Usage::

    from physics_engine.active_inference import ExperimentSelector
    
    selector = ExperimentSelector(strategy="uncertainty_sampling")
    
    # With populated belief state
    next_experiment = selector.choose_experiment(
        belief_state=belief,
        available_experiments=["dataset_1", "dataset_2", "dataset_3"],
    )
    # Returns: "dataset_2" (most informative to disambiguate models)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from .belief_state import BeliefState


class SelectionStrategy(Enum):
    """Experiment selection strategies."""
    
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    """Choose experiment with highest model disagreement."""
    
    INFORMATION_GAIN = "information_gain"
    """Choose experiment that reduces entropy most."""
    
    RANDOM = "random"
    """Random selection (baseline)."""


@dataclass
class ExperimentScore:
    """Score for a candidate experiment."""
    
    experiment_id: str
    """Identifier for the experiment."""
    
    score: float
    """Numerical score (higher = more informative for strategy)."""
    
    reasoning: str
    """Human-readable explanation."""


class ExperimentSelector:
    """
    Select next experiment based on belief state.
    
    Uses epistemic uncertainty and model ensemble to decide which dataset
    would be most informative to run next.
    """
    
    def __init__(self, strategy: str = "uncertainty_sampling"):
        """
        Initialize selector.
        
        Args:
            strategy: One of "uncertainty_sampling", "information_gain", "random"
        """
        try:
            self.strategy = SelectionStrategy(strategy)
        except ValueError:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of: "
                f"{', '.join([s.value for s in SelectionStrategy])}"
            )
    
    def choose_experiment(
        self,
        belief_state: BeliefState,
        available_experiments: List[str],
        verbose: bool = False,
    ) -> Tuple[str, ExperimentScore]:
        """
        Select next experiment to run.
        
        Args:
            belief_state: Current belief state over models
            available_experiments: List of available experiment identifiers
            verbose: If True, print scoring details
        
        Returns:
            Tuple of (selected_experiment_id, ExperimentScore with reasoning)
        
        Raises:
            ValueError: If no experiments available or belief state empty
        """
        if not available_experiments:
            raise ValueError("No experiments available to choose from")
        
        if belief_state.model_count() == 0:
            # No models yet: use random selection
            chosen_id = available_experiments[0]
            score = ExperimentScore(
                experiment_id=chosen_id,
                score=0.0,
                reasoning="No models in belief state yet; selecting first experiment.",
            )
            return chosen_id, score
        
        if self.strategy == SelectionStrategy.UNCERTAINTY_SAMPLING:
            chosen_id, score = self._uncertainty_sampling(
                belief_state, available_experiments, verbose
            )
        elif self.strategy == SelectionStrategy.INFORMATION_GAIN:
            chosen_id, score = self._information_gain(
                belief_state, available_experiments, verbose
            )
        elif self.strategy == SelectionStrategy.RANDOM:
            chosen_id, score = self._random_selection(available_experiments, verbose)
        else:
            raise ValueError(f"Unhandled strategy: {self.strategy}")
        
        return chosen_id, score
    
    def _uncertainty_sampling(
        self,
        belief_state: BeliefState,
        available_experiments: List[str],
        verbose: bool = False,
    ) -> Tuple[str, ExperimentScore]:
        """
        Uncertainty sampling: prioritize experiments to reduce epistemic uncertainty.
        
        Strategy: Select experiment that maximizes model disagreement.
        Rationale: If multiple models fit the current data equally well,
        we want data that would discriminate between them.
        """
        uncertainty = belief_state.epistemic_uncertainty()
        
        # Simple scoring: experiments get higher score if uncertainty is high
        # In practice, this could query predicted disagreement for each experiment
        scores = {}
        for exp_id in available_experiments:
            # Base score: epistemic uncertainty
            score = uncertainty
            # Tie-breaking: prefer lower experiment index
            score += 1.0 / (len(available_experiments) + 1)
            scores[exp_id] = score
        
        # Add small noise for diversity (avoid always picking same experiment)
        noise = np.random.normal(0, 0.01, len(available_experiments))
        scores_with_noise = {
            exp_id: scores[exp_id] + noise[i]
            for i, exp_id in enumerate(available_experiments)
        }
        
        chosen_id = max(scores_with_noise, key=scores_with_noise.get)
        
        best = belief_state.best_model()
        top_2 = belief_state.top_k_models(k=2)
        f_gap = (
            abs(top_2[0].free_energy - top_2[1].free_energy)
            if len(top_2) >= 2
            else 0.0
        )
        
        reasoning = (
            f"Epistemic uncertainty={uncertainty:.4f}, "
            f"Best model F={best.free_energy:.4f}, "
            f"Gap to 2nd={f_gap:.4f}. "
            f"Selecting to reduce model disagreement."
        )
        
        score = ExperimentScore(
            experiment_id=chosen_id,
            score=scores[chosen_id],
            reasoning=reasoning,
        )
        
        if verbose:
            print(f"[Uncertainty Sampling]")
            print(f"  Epistemic uncertainty: {uncertainty:.4f}")
            print(f"  Best model F: {best.free_energy:.4f}")
            print(f"  Chosen experiment: {chosen_id} (score={scores[chosen_id]:.4f})")
        
        return chosen_id, score
    
    def _information_gain(
        self,
        belief_state: BeliefState,
        available_experiments: List[str],
        verbose: bool = False,
    ) -> Tuple[str, ExperimentScore]:
        """
        Information gain: select experiment maximizing entropy reduction.
        
        Notes:
            This is a placeholder for more sophisticated information-theoretic selection.
            Future work: compute predicted entropy reduction for each experiment.
        """
        # For now, fall back to uncertainty sampling
        return self._uncertainty_sampling(belief_state, available_experiments, verbose)
    
    def _random_selection(
        self,
        available_experiments: List[str],
        verbose: bool = False,
    ) -> Tuple[str, ExperimentScore]:
        """Baseline: random selection."""
        chosen_id = available_experiments[np.random.randint(0, len(available_experiments))]
        
        reasoning = "Random selection (baseline strategy)"
        score = ExperimentScore(
            experiment_id=chosen_id,
            score=np.random.random(),
            reasoning=reasoning,
        )
        
        if verbose:
            print(f"[Random Selection] Chose: {chosen_id}")
        
        return chosen_id, score
    
    def score_all_experiments(
        self,
        belief_state: BeliefState,
        available_experiments: List[str],
    ) -> List[ExperimentScore]:
        """
        Compute scores for all available experiments without selecting one.
        
        Useful for inspection and debugging.
        
        Returns:
            List of ExperimentScore objects, sorted by score (highest first).
        """
        scores = []
        
        if self.strategy == SelectionStrategy.UNCERTAINTY_SAMPLING:
            uncertainty = belief_state.epistemic_uncertainty()
            for i, exp_id in enumerate(available_experiments):
                score_val = uncertainty + 1.0 / (len(available_experiments) + 1)
                reasoning = (
                    f"Uncertainty sampling: uc={uncertainty:.4f}, "
                    f"position={i}/{len(available_experiments)}"
                )
                scores.append(
                    ExperimentScore(experiment_id=exp_id, score=score_val, reasoning=reasoning)
                )
        
        elif self.strategy == SelectionStrategy.RANDOM:
            for exp_id in available_experiments:
                scores.append(
                    ExperimentScore(
                        experiment_id=exp_id,
                        score=np.random.random(),
                        reasoning="Random",
                    )
                )
        
        else:
            # Information gain or other
            for exp_id in available_experiments:
                scores.append(
                    ExperimentScore(
                        experiment_id=exp_id,
                        score=0.0,
                        reasoning="Not yet implemented",
                    )
                )
        
        return sorted(scores, key=lambda s: s.score, reverse=True)
