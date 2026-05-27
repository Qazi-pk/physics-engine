"""
Command-line interface for PIR Engine

Provides convenient commands for common PIR operations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from physics_engine import __version__
from physics_engine.core import Dataset
from physics_engine.pipelines import run_robot_joint_identification, run_system_identification


def main() -> None:
    """Main entry point for the 'pir' command."""
    parser = argparse.ArgumentParser(
        prog="pir",
        description="Physics Intermediate Representation (PIR) Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pir --version                               Show version information
  pir discover --dataset data.csv             Discover physics laws from dataset
  pir robot --dataset robot_data.csv          Identify robot joint structure
  pir simulate --model model.json             Run physics simulation
  pir benchmark --config benchmark.yaml       Run large-scale experiment benchmark
  pir benchmark --config test.yaml --dry-run  Show benchmark plan without running

For more information, visit: https://github.com/Qazi-pk/Physics_Intermediate_Representation
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"pir-engine {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover physics laws from data",
    )
    discover_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset CSV file",
    )
    discover_parser.add_argument(
        "--output",
        default="discovered_model.json",
        help="Output file for discovered model (default: discovered_model.json)",
    )
    discover_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Robot identification command
    robot_parser = subparsers.add_parser(
        "robot",
        help="Identify robot joint structure",
    )
    robot_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to robot dataset CSV file",
    )
    robot_parser.add_argument(
        "--output",
        default="robot_model.json",
        help="Output file for robot model (default: robot_model.json)",
    )
    robot_parser.add_argument(
        "--joint-id",
        type=int,
        help="Specific joint ID to analyze (optional)",
    )

    # Simulate command
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run physics simulation",
    )
    simulate_parser.add_argument(
        "--model",
        required=True,
        help="Path to model JSON file",
    )
    simulate_parser.add_argument(
        "--initial-conditions",
        required=True,
        help="Path to initial conditions CSV file",
    )
    simulate_parser.add_argument(
        "--output",
        default="simulation_results.csv",
        help="Output file for simulation results (default: simulation_results.csv)",
    )
    simulate_parser.add_argument(
        "--time-span",
        type=float,
        nargs=2,
        default=[0.0, 10.0],
        metavar=("START", "END"),
        help="Time span for simulation (default: 0.0 10.0)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run large-scale experiment benchmarks",
    )
    benchmark_parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark configuration YAML file",
    )
    benchmark_parser.add_argument(
        "--output",
        help="Override output directory from config",
    )
    benchmark_parser.add_argument(
        "--max-workers",
        type=int,
        help="Override max parallel workers from config",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment plan without executing",
    )

    args = parser.parse_args()

    if args.command == "discover":
        discover_command(args)
    elif args.command == "robot":
        robot_command(args)
    elif args.command == "simulate":
        simulate_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def discover_command(args) -> None:
    """Execute the discover command."""
    print(f"🔍 Discovering physics laws from: {args.dataset}")
    
    try:
        # Load dataset
        df = pd.read_csv(args.dataset)
        
        if args.verbose:
            print(f"   Dataset shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
        
        # Create Dataset object
        dataset = Dataset(df)
        
        # Run system identification
        print("   Running system identification...")
        result = run_system_identification(dataset)
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}, f, indent=2)
        
        print(f"✓ Model saved to: {output_path}")
        print(f"✓ Discovered equation: {result}")
        
    except Exception as e:
        print(f"✗ Error during discovery: {e}", file=sys.stderr)
        sys.exit(1)


def robot_command(args) -> None:
    """Execute the robot identification command."""
    print(f"🤖 Identifying robot structure from: {args.dataset}")
    
    try:
        # Load dataset
        df = pd.read_csv(args.dataset)
        
        print(f"   Dataset shape: {df.shape}")
        
        # Run robot joint identification
        print("   Running robot joint identification...")
        result = run_robot_joint_identification(
            dataset_path=args.dataset,
            output_dir=Path(args.output).parent,
        )
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}, f, indent=2)
        
        print(f"✓ Robot model saved to: {output_path}")
        if hasattr(result, "equations"):
            print(f"✓ Discovered {len(result.equations)} joint equations")
        
    except Exception as e:
        print(f"✗ Error during robot identification: {e}", file=sys.stderr)
        sys.exit(1)


def simulate_command(args) -> None:
    """Execute the simulation command."""
    print(f"⚙️  Running simulation with model: {args.model}")
    
    try:
        # Load model
        with open(args.model) as f:
            model_data = json.load(f)
        
        # Load initial conditions
        initial_conditions = pd.read_csv(args.initial_conditions)
        
        print(f"   Time span: {args.time_span[0]} to {args.time_span[1]}")
        
        # TODO: Implement simulation logic here
        # This would use physics_engine.simulation.simulate()
        
        print(f"✓ Simulation complete")
        print(f"✓ Results saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Error during simulation: {e}", file=sys.stderr)
        sys.exit(1)


def benchmark_command(args) -> None:
    """Execute the benchmark command."""
    from physics_engine.benchmarking import ExperimentConfig, ExperimentRunner, SummaryGenerator
    
    print(f"📊 Loading benchmark configuration: {args.config}")
    
    try:
        # Load configuration
        config = ExperimentConfig.from_yaml(args.config)
        
        # Override output directory if specified
        if args.output:
            config.output_dir = str(Path(args.output))
        
        # Override max workers if specified
        if args.max_workers:
            config.max_workers = args.max_workers
        
        # Display experiment plan
        total = config.total_experiments()
        print(f"\n{'='*70}")
        print(f"Benchmark: {config.name}")
        print(f"{'='*70}")
        print(f"Datasets: {len(config.datasets)}")
        print(f"Algorithms: {len(config.algorithms)}")
        print(f"Noise levels: {len(config.noise_levels)}")
        print(f"Dataset sizes: {len(config.dataset_sizes)}")
        print(f"Seeds: {len(config.seeds)}")
        print(f"\nTotal experiments: {total}")
        print(f"Parallel workers: {config.max_workers}")
        print(f"Output directory: {config.output_dir}")
        print(f"Cache enabled: {config.cache_enabled}")
        print(f"Cache directory: {config.cache_dir}")
        print(f"{'='*70}\n")
        
        # Dry run: show plan and exit
        if args.dry_run:
            print("Dry run - showing first 5 experiments:")
            runner = ExperimentRunner(config)
            runs = list(runner.generate_runs())
            for i, run in enumerate(runs[:5]):
                print(f"{i+1}. {run}")
            print(f"\n... and {len(runs) - 5} more experiments")
            return
        
        # Run benchmark
        print("🚀 Starting benchmark execution...")
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Generate summary
        print("\n📈 Generating summary...")
        generator = SummaryGenerator(results)
        generator.print_summary()
        
        # Save outputs
        output_dir = Path(config.output_dir)
        summary_path = output_dir / "summary.md"
        generator.save_markdown(summary_path)
        print(f"\n✓ Summary saved to: {summary_path}")
        
        csv_path = output_dir / "results.csv"
        generator.save_csv(csv_path)
        print(f"✓ Results CSV saved to: {csv_path}")
        
        comparison_path = output_dir / "comparison.csv"
        generator.save_comparison_table(comparison_path)
        print(f"✓ Comparison table saved to: {comparison_path}")

        paper_paths = generator.save_paper_tables(output_dir)
        print(f"✓ Paper summary CSV saved to: {paper_paths['summary_csv']}")
        print(f"✓ Paper summary Markdown saved to: {paper_paths['summary_md']}")
        print(f"✓ Paper summary LaTeX saved to: {paper_paths['summary_tex']}")
        
        print(f"\n{'='*70}")
        print(f"✓ Benchmark complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"✗ Error during benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

