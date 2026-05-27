"""
belief_state.py
~~~~~~~~~~~~~~~

Maintain a belief state over discovered models.

The belief state stores discovered models with their uncertainties (free energies)
and provides operations to:
  - Add new candidate laws/models
  - Retrieve best model(s) based on free energy
  - Extract epistemic uncertainty (model ensemble variance)
  - Update confidence based on new evidence

Usage::

    from physics_engine.active_inference import BeliefState
    
    belief = BeliefState()
    
    # After discovery cycle 1: discovered law1 with free energy F1
    belief.add_model("law1", discovered_law_dict, free_energy=F1, n_params=8)
    
    # After discovery cycle 2: discovered law2 with free energy F2
    belief.add_model("law2", discovered_law_dict2, free_energy=F2, n_params=10)
    
    # Get best model
    best_model = belief.best_model()
    
    # Get uncertainty (how different are the top models?)
    uncertainty = belief.epistemic_uncertainty()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ModelCandidate:
    """A single candidate model in the belief state."""
    
    id: str
    """Unique identifier for this model."""
    
    law_dict: Dict[str, Any]
    """The discovered law (equation, coefficients, etc.)."""
    
    free_energy: float
    """Free energy (lower is better)."""
    
    n_params: int
    """Number of free parameters."""
    
    mse: float
    """Mean squared error on training data."""
    
    confidence: float
    """Confidence score (1.0 = highest, 0.0 = lowest)."""
    
    evidence_count: int = 1
    """How many times this model has been validated."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (algorithm, dataset, etc.)."""
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if self.free_energy < 0:
            raise ValueError(f"free_energy must be >= 0, got {self.free_energy}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


class BeliefState:
    """
    Belief state over discovered models.
    
    Maintains a repository of candidate models with their uncertainties.
    Enables model selection based on free energy and evidence aggregation.
    """
    
    def __init__(self):
        """Initialize empty belief state."""
        self.models: Dict[str, ModelCandidate] = {}
        self.update_count = 0
    
    def add_model(
        self,
        model_id: str,
        law_dict: Dict[str, Any],
        free_energy: float,
        n_params: int,
        mse: float,
        confidence: float = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Add a candidate model to the belief state.
        
        Args:
            model_id: Unique identifier (often timestamp or cycle index)
            law_dict: The discovered law (equation, coefficients, variables, domain, etc.)
            free_energy: Free energy score (lower is better)
            n_params: Number of fitted parameters
            mse: Mean squared error
            confidence: (Optional) Confidence in this model (default: compute from F)
            metadata: (Optional) Additional info (algorithm, dataset, date, etc.)
        
        Notes:
            If confidence not provided, it's computed as exp(-F/T) where T=1.0
            (Boltzmann-like temperature parameter).
        """
        if model_id in self.models:
            # Update existing model
            self.models[model_id].evidence_count += 1
            self.models[model_id].law_dict = law_dict
            self.models[model_id].free_energy = free_energy
            self.models[model_id].mse = mse
        else:
            # New model
            if confidence is None:
                # Boltzmann-like confidence from free energy
                temperature = 1.0
                confidence = float(np.exp(-free_energy / temperature))
                confidence = min(confidence, 1.0)  # Clamp to [0,1]
            
            model = ModelCandidate(
                id=model_id,
                law_dict=law_dict,
                free_energy=free_energy,
                n_params=n_params,
                mse=mse,
                confidence=confidence,
                metadata=metadata or {},
            )
            self.models[model_id] = model
        
        self.update_count += 1
    
    def best_model(self) -> Optional[ModelCandidate]:
        """
        Get the single best model (lowest free energy).
        
        Returns:
            ModelCandidate with lowest F, or None if no models yet.
        """
        if not self.models:
            return None
        return min(self.models.values(), key=lambda m: m.free_energy)
    
    def top_k_models(self, k: int = 3) -> List[ModelCandidate]:
        """
        Get top k models ranked by free energy.
        
        Args:
            k: Number of models to return
        
        Returns:
            List of ModelCandidate objects, sorted by F (lowest first).
        """
        sorted_models = sorted(self.models.values(), key=lambda m: m.free_energy)
        return sorted_models[:k]
    
    def epistemic_uncertainty(self) -> float:
        """
        Compute epistemic uncertainty = variance in top models.
        
        High uncertainty means multiple competing models;
        low uncertainty means consensus on one model.
        
        Returns:
            Scalar uncertainty score (0 = certain, >0 = uncertain).
        
        Notes:
            Computed as variance of free energies across top 3 models.
            If fewer than 3 models, returns 0 (certain).
        """
        if len(self.models) < 2:
            return 0.0
        
        top_models = self.top_k_models(k=3)
        free_energies = [m.free_energy for m in top_models]
        
        if len(free_energies) < 2:
            return 0.0
        
        uncertainty = float(np.var(free_energies))
        return uncertainty
    
    def aleatoric_uncertainty(self) -> float:
        """
        Compute aleatoric (data) uncertainty = MSE of best model.
        
        Returns:
            MSE of the best model.
        """
        best = self.best_model()
        if best is None:
            return 0.0
        return best.mse
    
    def total_uncertainty(self) -> float:
        """
        Compute total uncertainty = epistemic + aleatoric.
        
        Returns:
            Sum of epistemic and aleatoric uncertainties.
        """
        return self.epistemic_uncertainty() + self.aleatoric_uncertainty()
    
    def get_model(self, model_id: str) -> Optional[ModelCandidate]:
        """
        Retrieve a specific model by ID.
        
        Args:
            model_id: The model identifier
        
        Returns:
            ModelCandidate or None if not found.
        """
        return self.models.get(model_id)
    
    def model_count(self) -> int:
        """Return number of models in belief state."""
        return len(self.models)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize belief state to dictionary.
        
        Returns:
            Dictionary with models, uncertainties, and metadata.
        """
        return {
            "models": {
                mid: {
                    "law": m.law_dict,
                    "free_energy": m.free_energy,
                    "n_params": m.n_params,
                    "mse": m.mse,
                    "confidence": m.confidence,
                    "evidence_count": m.evidence_count,
                    "metadata": m.metadata,
                }
                for mid, m in self.models.items()
            },
            "best_model_id": self.best_model().id if self.best_model() else None,
            "epistemic_uncertainty": self.epistemic_uncertainty(),
            "aleatoric_uncertainty": self.aleatoric_uncertainty(),
            "total_uncertainty": self.total_uncertainty(),
            "update_count": self.update_count,
        }
