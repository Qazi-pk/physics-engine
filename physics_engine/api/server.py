from fastapi import FastAPI, HTTPException

from .schemas import AskInput, AskOutput, ProblemInput, SolutionOutput
from ..core.control_engine import ControlEngine
from ..interface import answer_physics_question
from ..laws.registry import registry


# ------------------------------------------------------------------
# App definition
# ------------------------------------------------------------------

app = FastAPI(
    title="Physics Engine API",
    description="Symbolic + Numeric Physics Solver with Physical Constraints",
    version="0.1.0"
)


# ------------------------------------------------------------------
# API endpoint
# ------------------------------------------------------------------

@app.post("/solve", response_model=SolutionOutput)
def solve_problem(problem: ProblemInput):
    """
    Solve a physics problem given a natural language description
    and optional known values.
    """
    try:
        engine = ControlEngine(description=problem.description)

        if problem.known_values:
            for symbol_name, value in problem.known_values.items():
                engine.add_known_value(symbol_name, value)

        result, metadata = engine.route(registry.list())

        return SolutionOutput(
            laws=result.get("laws", []),
            equations=[str(eq) for eq in result.get("equations", [])],
            known_values={str(k): v for k, v in result.get("known_values", {}).items()},
            solution={str(k): v for k, v in result.get("solution", {}).items()},
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskOutput)
def ask_physics(problem: AskInput):
    """
    Answer a natural-language physics question via knowledge lookup
    or data-driven discovery when dataset inputs are provided.
    """
    try:
        response = answer_physics_question(
            question=problem.question,
            dataset_path=problem.dataset_path,
            target_var=problem.target_var,
            domain=problem.domain,
            run_discovery=problem.run_discovery,
            run_scientist_loop=problem.run_scientist_loop,
            scientist_cycles=problem.scientist_cycles,
            scientist_discovery_mode=problem.scientist_discovery_mode,
            scientist_discover_kwargs=problem.scientist_discover_kwargs,
            scientist_use_active_inference=problem.scientist_use_active_inference,
            scientist_active_inference_strategy=problem.scientist_active_inference_strategy,
            scientist_available_experiments=problem.scientist_available_experiments,
        )
        return AskOutput(result=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
