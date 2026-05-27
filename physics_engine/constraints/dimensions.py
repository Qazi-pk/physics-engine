from dataclasses import dataclass


class DimensionError(Exception):
    """
    Raised when a dimensional inconsistency is detected.
    """
    pass


@dataclass(frozen=True)
class Dimension:
    """
    Physical dimension represented as exponents of base dimensions.

    Default base dimensions:
    - M : Mass
    - L : Length
    - T : Time

    Example:
        Force = M^1 L^1 T^-2
    """
    M: float = 0.0
    L: float = 0.0
    T: float = 0.0

    # ----------------------------
    # Arithmetic operations
    # ----------------------------

    def __add__(self, other):
        """
        Addition is ONLY allowed for identical dimensions.
        """
        if self != other:
            raise DimensionError(
                f"Cannot add incompatible dimensions: {self} and {other}"
            )
        return self

    def __sub__(self, other):
        """
        Subtraction follows the same rule as addition.
        """
        if self != other:
            raise DimensionError(
                f"Cannot subtract incompatible dimensions: {self} and {other}"
            )
        return self

    def __mul__(self, other):
        """
        Multiplication combines dimensions.
        """
        if not isinstance(other, Dimension):
            raise TypeError("Can only multiply Dimension by Dimension")

        return Dimension(
            M=self.M + other.M,
            L=self.L + other.L,
            T=self.T + other.T,
        )

    def __truediv__(self, other):
        """
        Division subtracts dimension exponents.
        """
        if not isinstance(other, Dimension):
            raise TypeError("Can only divide Dimension by Dimension")

        return Dimension(
            M=self.M - other.M,
            L=self.L - other.L,
            T=self.T - other.T,
        )

    def __pow__(self, power):
        """
        Power scales dimension exponents.
        """
        if not isinstance(power, (int, float)):
            raise DimensionError(
                f"Invalid power '{power}' for dimension exponentiation"
            )

        return Dimension(
            M=self.M * power,
            L=self.L * power,
            T=self.T * power,
        )

    # ----------------------------
    # Comparison & representation
    # ----------------------------

    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False

        return (
            self.M == other.M
            and self.L == other.L
            and self.T == other.T
        )

    def is_dimensionless(self) -> bool:
        """
        Returns True if the quantity has no physical dimension.
        """
        return self.M == 0 and self.L == 0 and self.T == 0

    def __repr__(self):
        parts = []

        if self.M != 0:
            parts.append(f"M^{self.M}")
        if self.L != 0:
            parts.append(f"L^{self.L}")
        if self.T != 0:
            parts.append(f"T^{self.T}")

        if not parts:
            return "Dimensionless"

        return " ".join(parts)
