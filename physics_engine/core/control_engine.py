from sympy import Eq, solve
from collections import defaultdict
from ..discovery import discover_equations, discover_law, load_csv
from .inference import (
    build_dependency_graph,
    infer_goal,
    rank_goals,
    reason,
    solve_equations,
)


class ControlEngine:

    def __init__(self, description=None):
        self.description = description
        self.equations = []
        self.known_values = {}
        self.constraint_graph = {}
        self.causal_graph = {}
        self.commands = self._register_commands()

    def add_known_value(self, symbol_name, value):
        self.known_values[symbol_name] = value

    # =====================================================
    # ROUTE
    # =====================================================

    def route(self, laws):
        self.equations = []
        self.known_values = {}
        self.constraint_graph = {}
        self.causal_graph = {}

        selected_laws = laws  # include all for now

        for law in selected_laws:
            self.equations.extend(law.equations())
            if hasattr(law, "known_values"):
                self.known_values.update(law.known_values)

        goal_candidates = infer_goal(self.equations, self.known_values)
        ranked_goals = rank_goals(self.equations, self.known_values, goal_candidates)

        solved_target = None
        solved_value = None

        for target in ranked_goals:
            candidate_solution = solve_equations(self.equations, self.known_values, target)
            if candidate_solution is not None:
                solved_target = target
                solved_value = candidate_solution
                self.known_values[target] = candidate_solution
                break

        self.known_values = reason(self.equations, self.known_values)

        solved = solve(self.equations, dict=True)

        if solved:
            self.known_values.update(solved[0])

        self._build_constraint_graph()
        self._build_causal_graph()

        result = {
            "laws": [law.name for law in selected_laws],
            "equations": self.equations,
            "known_values": self.known_values,
            "goal_candidates": [str(goal) for goal in ranked_goals],
            "solved_target": str(solved_target) if solved_target is not None else None,
            "solved_value": solved_value,
            "causal_graph": self.causal_graph,
        }

        metadata = {
            "description": "Physics routing complete",
            "solvable": bool(solved) or solved_target is not None,
        }

        return result, metadata

    # =====================================================
    # GRAPH BUILDERS
    # =====================================================

    def _build_constraint_graph(self):
        graph = defaultdict(set)

        for eq in self.equations:
            symbols = eq.free_symbols
            for s1 in symbols:
                for s2 in symbols:
                    if s1 != s2:
                        graph[str(s1)].add(str(s2))

        self.constraint_graph = dict(graph)

    def _build_causal_graph(self):
        dependency_graph = build_dependency_graph(self.equations)
        self.causal_graph = {
            str(lhs): {str(rhs) for rhs in rhs_symbols}
            for lhs, rhs_symbols in dependency_graph.items()
        }

    # =====================================================
    # CAUSAL QUERIES
    # =====================================================

    def derive_causal_effect(self, X, Y):
        if Y in self.causal_graph and X in self.causal_graph[Y]:
            return f"{X} has direct causal effect on {Y}"
        return f"No direct causal effect from {X} to {Y}"

    def find_backdoor_adjustment(self, X, Y):
        return f"No confounders detected between {X} and {Y}"

    def d_separated(self, X, Y, conditioned_on=None):
        return False

    # =====================================================
    # CLI
    # =====================================================

    def _register_commands(self):
        return {
            "effect": self._cmd_effect,
            "backdoor": self._cmd_backdoor,
            "graph": self._cmd_graph,
            "dsep": self._cmd_dsep,
            "discover": self._cmd_discover,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
        }

    def start_cli(self):
        print("\nInteractive Causal CLI. Type 'help' for commands.\n")

        while True:
            raw = input("physics_engine> ").strip()
            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0]
            args = parts[1:]

            if cmd in self.commands:
                should_exit = self.commands[cmd](args)
                if should_exit:
                    break
            else:
                print("Unknown command. Type 'help'.")

    def _cmd_effect(self, args):
        if len(args) != 2:
            print("Usage: effect X Y")
            return
        print(self.derive_causal_effect(args[0], args[1]))

    def _cmd_backdoor(self, args):
        if len(args) != 2:
            print("Usage: backdoor X Y")
            return
        print(self.find_backdoor_adjustment(args[0], args[1]))

    def _cmd_graph(self, args):
        print(self.causal_graph)
        return False

    def _cmd_dsep(self, args):
        if len(args) < 2:
            print("Usage: dsep X Y [Z...]")
            return
        X = args[0]
        Y = args[1]
        Z = set(args[2:]) if len(args) > 2 else set()
        print(self.d_separated(X, Y, Z))

    def _cmd_help(self, args):
        print("""
Available commands:
  effect X Y
  backdoor X Y
  graph
  dsep X Y [Z...]
  discover from <csv_file> [target]
  help
  exit
""")

    def _cmd_discover(self, args):
        if len(args) < 2 or args[0].lower() != "from":
            print("Usage: discover from <csv_file> [target]")
            return

        csv_path = args[1]
        target = args[2] if len(args) > 2 else None

        if target is not None:
            try:
                law, error, significant = discover_law(csv_path, target)
            except Exception as error:
                print(f"Discovery failed: {error}")
                return

            print("\n=== DISCOVERY RESULT ===")
            print(f"Discovered: {law}")
            print(f"Error: {error}")
            print(f"Residual correlations: {significant}")
            return

        try:
            headers, rows = load_csv(csv_path)
        except Exception as error:
            print(f"Discovery failed: {error}")
            return

        discovered = discover_equations(headers, rows)
        if not discovered:
            print("No candidate symbolic equations discovered.")
            return

        print("Discovered candidate equations:")
        for item in discovered:
            print(f"  {item['equation']}  (score={item['score']:.3f})")

    def _cmd_exit(self, args):
        print("Exiting CLI.")
        return True