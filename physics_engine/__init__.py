# physics_engine package
"""
Physics Intermediate Representation (PIR) Engine
A comprehensive framework for automated physics law discovery from data.

Submodules are imported lazily: `import physics_engine` does NOT eagerly load
pipelines/benchmarks/variational/etc. This keeps lightweight consumer imports
(e.g. `from physics_engine.sklearn_adapter import PIRRegressor`) from requiring
the entire engine to be present. Access submodules explicitly, e.g.
`from physics_engine.pipelines import run_system_identification`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
