from .experiment_designer import ExperimentProposal, design_next_experiment
from .hypothesis_generator import generate_guided_hypotheses
from .scientist_loop import AIScientist, ScientistCycleResult
from .theory_manager import Theory, TheoryManager

__all__ = [
    "AIScientist",
    "ScientistCycleResult",
    "ExperimentProposal",
    "Theory",
    "TheoryManager",
    "generate_guided_hypotheses",
    "design_next_experiment",
]
