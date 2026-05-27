from sympy import symbols, Eq
from ..constraints.dimensions import Dimension


class NewtonSecondLaw:
    name = "Newton Second Law"

    def __init__(self):
        self.F, self.m, self.a = symbols("F m a")

    def applicable(self, metadata) -> bool:
        return getattr(metadata, "domain", None) == "mechanics"

    def equations(self):
        return [Eq(self.F, self.m * self.a)]

    def symbols(self):
        return {self.F, self.m, self.a}

    def dimensions(self):
        """
        Canonical physical dimensions for symbols in this law.
        """
        return {
            self.F: Dimension(M=1, L=1, T=-2),
            self.m: Dimension(M=1),
            self.a: Dimension(L=1, T=-2),
        }


class KinematicsVelocityLaw:
    name = "Velocity from constant acceleration"

    def __init__(self):
        self.v, self.u, self.a, self.t = symbols("v u a t")

    def applicable(self, metadata) -> bool:
        return getattr(metadata, "domain", None) == "mechanics"

    def equations(self):
        return [Eq(self.v, self.u + self.a * self.t)]

    def symbols(self):
        return {self.v, self.u, self.a, self.t}

    def dimensions(self):
        return {
            self.v: Dimension(L=1, T=-1),
            self.u: Dimension(L=1, T=-1),
            self.a: Dimension(L=1, T=-2),
            self.t: Dimension(T=1),
        }


class KineticEnergyLaw:
    name = "Kinetic Energy"

    def __init__(self):
        self.E, self.m, self.v = symbols("E m v")

    def applicable(self, metadata) -> bool:
        return getattr(metadata, "domain", None) == "mechanics"

    def equations(self):
        return [Eq(self.E, 0.5 * self.m * self.v**2)]

    def symbols(self):
        return {self.E, self.m, self.v}

    def dimensions(self):
        return {
            self.E: Dimension(M=1, L=2, T=-2),
            self.m: Dimension(M=1),
            self.v: Dimension(L=1, T=-1),
        }
