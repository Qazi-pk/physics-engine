from importlib import import_module

try:
    from .core.control_engine import ControlEngine
    from .laws import registry
except ImportError:
    ControlEngine = import_module("physics_engine.core.control_engine").ControlEngine
    registry = import_module("physics_engine.laws").registry


def main():
    engine = ControlEngine()

    laws = registry.list()
    print([law.name for law in laws])

    result, metadata = engine.route(laws)

    print("\n=== RESULT ===")
    print(result)

    print("\n=== METADATA ===")
    print(metadata)


    engine.start_cli()


if __name__ == "__main__":
    from experiments.kepler_third_law import run_kepler_experiment

    print("Running Physics Engine Experiments")
    run_kepler_experiment()
    main()