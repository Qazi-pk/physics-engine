from __future__ import annotations

from typing import Callable


class PhysicsModel:
    """
    Generic representation of a physical dynamical system.
    """

    def __init__(self, variables: list[str], parameters: dict, equation_function: Callable):
        self.variables = list(variables)
        self.parameters = dict(parameters)
        self.equation_function = equation_function

    def derivatives(self, t, state):
        return self.equation_function(state, t, self.parameters)

    def describe(self) -> dict:
        return {
            "variables": self.variables,
            "parameters": self.parameters,
        }
