from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import sympy as sp

from .representation import Equation


def _parse_dimensions(dimensions: Any) -> Any:
    if dimensions is None:
        return None
    if isinstance(dimensions, (dict, str)):
        return dimensions
    return str(dimensions)


def equation_to_ir_dict(equation: Equation) -> Dict[str, Any]:
    rhs_expr = sp.sympify(equation.rhs)
    lhs_expr = sp.sympify(equation.lhs)
    target = str(lhs_expr)
    variables = sorted(str(sym) for sym in rhs_expr.free_symbols if str(sym) != target)

    payload: Dict[str, Any] = {
        "type": "equation",
        "target": target,
        "expression": str(rhs_expr),
        "variables": variables,
        "dimensions": _parse_dimensions(equation.dimensions),
        "regime": equation.regime,
        "metadata": dict(equation.metadata),
    }
    return payload


def equation_from_ir_dict(payload: Dict[str, Any]) -> Equation:
    target = payload.get("target", payload.get("lhs"))
    expression = payload.get("expression", payload.get("rhs"))

    if target is None or expression is None:
        raise ValueError("payload must include target/expression or lhs/rhs")

    return Equation(
        lhs=str(target),
        rhs=str(expression),
        dimensions=payload.get("dimensions"),
        regime=payload.get("regime"),
        metadata={str(k): str(v) for k, v in dict(payload.get("metadata", {})).items()},
    )


def dump_equation_json(equation: Equation, indent: int = 2) -> str:
    return json.dumps(equation_to_ir_dict(equation), indent=indent, sort_keys=True)


def load_equation_json(payload: str) -> Equation:
    return equation_from_ir_dict(json.loads(payload))


def write_pir_json(equation: Equation, path: Union[str, Path], indent: int = 2) -> Path:
    output_path = Path(path)
    output_path.write_text(dump_equation_json(equation, indent=indent), encoding="utf-8")
    return output_path


def read_pir_json(path: Union[str, Path]) -> Equation:
    input_path = Path(path)
    return load_equation_json(input_path.read_text(encoding="utf-8"))
