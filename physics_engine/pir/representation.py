from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union


@dataclass(frozen=True)
class Equation:
    lhs: str
    rhs: str
    dimensions: Optional[Union[str, Dict[str, Any]]] = None
    regime: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lhs": self.lhs,
            "rhs": self.rhs,
            "dimensions": self.dimensions,
            "regime": self.regime,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PIRStateVariable:
    name: str
    dimension: Optional[str] = None

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PIRDerivative:
    variable: Union[str, PIRStateVariable]
    order: int = 1
    respect_to: str = "t"

    def __post_init__(self):
        if self.order < 1:
            raise ValueError("Derivative order must be >= 1")

    @property
    def variable_name(self) -> str:
        if isinstance(self.variable, PIRStateVariable):
            return self.variable.name
        return str(self.variable)

    def __repr__(self) -> str:
        variable_name = self.variable_name
        if self.order == 1:
            return f"d{variable_name}/d{self.respect_to}"
        return f"d^{self.order}{variable_name}/d{self.respect_to}^{self.order}"


@dataclass(frozen=True)
class PIRDifferentialEquation:
    lhs: object
    rhs: object
    order: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "lhs": str(self.lhs),
            "rhs": str(self.rhs),
            "order": self.order,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PIRVectorVariable:
    name: str
    dimension: Optional[str] = None
    size: Optional[int] = None

    def __post_init__(self):
        if self.size is not None and self.size < 1:
            raise ValueError("Vector size must be >= 1")

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PIRMatrixVariable:
    name: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    dimension: Optional[str] = None

    def __post_init__(self):
        if self.rows is not None and self.rows < 1:
            raise ValueError("Matrix rows must be >= 1")
        if self.cols is not None and self.cols < 1:
            raise ValueError("Matrix cols must be >= 1")

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PIRPartialDerivative:
    variable: Union[str, PIRStateVariable, PIRVectorVariable]
    respect_to: Union[str, PIRStateVariable, PIRVectorVariable]

    def _as_name(self, value: Union[str, PIRStateVariable, PIRVectorVariable]) -> str:
        if isinstance(value, (PIRStateVariable, PIRVectorVariable)):
            return value.name
        return str(value)

    def __repr__(self) -> str:
        return f"∂{self._as_name(self.variable)}/∂{self._as_name(self.respect_to)}"


@dataclass(frozen=True)
class PIRMatrixEquation:
    lhs: Union[str, PIRMatrixVariable]
    rhs: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.lhs} = {self.rhs}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "lhs": str(self.lhs),
            "rhs": self.rhs,
            "metadata": dict(self.metadata),
        }


class PIROperators:
    @staticmethod
    def matrix_multiply(*terms: Union[str, PIRMatrixVariable]) -> str:
        if len(terms) < 2:
            raise ValueError("matrix_multiply requires at least two terms")
        return " @ ".join(str(term) for term in terms)

    @staticmethod
    def partial_derivative(
        variable: Union[str, PIRStateVariable, PIRVectorVariable],
        respect_to: Union[str, PIRStateVariable, PIRVectorVariable],
    ) -> PIRPartialDerivative:
        return PIRPartialDerivative(variable=variable, respect_to=respect_to)


@dataclass(frozen=True)
class PIRHamiltonian:
    coordinates: Sequence[str]
    momenta: Sequence[str]
    expression: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.coordinates) == 0 or len(self.momenta) == 0:
            raise ValueError("Hamiltonian requires non-empty coordinates and momenta")
        if len(self.coordinates) != len(self.momenta):
            raise ValueError("coordinates and momenta must have the same length")

    def __repr__(self) -> str:
        return f"H({', '.join(self.coordinates + self.momenta)}) = {self.expression}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "coordinates": list(self.coordinates),
            "momenta": list(self.momenta),
            "expression": self.expression,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PIRLagrangian:
    coordinates: Sequence[str]
    velocities: Sequence[str]
    expression: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.coordinates) == 0 or len(self.velocities) == 0:
            raise ValueError("Lagrangian requires non-empty coordinates and velocities")
        if len(self.coordinates) != len(self.velocities):
            raise ValueError("coordinates and velocities must have the same length")

    def __repr__(self) -> str:
        return f"L({', '.join(self.coordinates + self.velocities)}) = {self.expression}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "coordinates": list(self.coordinates),
            "velocities": list(self.velocities),
            "expression": self.expression,
            "metadata": dict(self.metadata),
        }


@dataclass
class PIRSystem:
    equations: Sequence[Union[PIRDifferentialEquation, str]]

    def describe(self) -> None:
        for equation in self.equations:
            print(equation)
