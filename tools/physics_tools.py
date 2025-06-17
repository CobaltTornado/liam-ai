"""physics_tool_v2_cleaned.py
--------------------------------------------------
Symbolic‑ and vector‑math helpers for the agentic tool‑chain.
This version removes merge‑conflict artefacts, unifies duplicate blocks, and
adds full type hints + defensive logging.

Public API
~~~~~~~~~~
* ``symbolic_manipulation`` – solve/differentiate/integrate/simplify/etc.
* ``vector_operation``       – dot, cross, scalar multiplication, magnitude,
                               normalization.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from sympy import (
    Matrix,
    SympifyError,
    diff,
    integrate,
    latex,
    solve,
    sympify,
    symbols,
)

__all__ = ["symbolic_manipulation", "vector_operation"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
PHYSICS_LOGGER = logging.getLogger("PhysicsToolV2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_return_dict(
    status: str,
    *,
    result: Any | None = None,
    reason: str | None = None,
    latex_repr: str | None = None,
    reasoning: str | None = None,
) -> Dict[str, Any]:
    """Return a standard response payload."""
    payload: Dict[str, Any] = {"status": status}

    if result is not None:
        if hasattr(result, "tolist"):
            payload["result"] = result.tolist()  # Matrix → list‑of‑lists
        elif isinstance(result, list):
            payload["result"] = [str(item) for item in result]
        else:
            payload["result"] = str(result)

    if reason:
        payload["reason"] = reason
    if latex_repr:
        payload["latex_representation"] = latex_repr
    if reasoning:
        payload["reasoning"] = reasoning
    return payload

# ---------------------------------------------------------------------------
# Symbolic manipulation
# ---------------------------------------------------------------------------

def symbolic_manipulation(
    expression: str,
    operation: str,
    *,
    variables: str = "",
    solve_for: Optional[str] = None,
    wrt: Optional[str] = None,
    at_point: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Perform high‑level symbolic math operations.

    Parameters
    ----------
    expression : str
        The expression/equation (e.g. ``"x**2 + y = z"``).
    operation : str
        One of ``solve | diff | integrate | simplify | assign | subs``.
    variables : str, optional
        Comma‑separated list of symbols present in *expression*.
    solve_for : str, optional
        Target symbol when *operation* = ``solve``.
    wrt : str, optional
        Variable of differentiation/integration ("with respect to").
    at_point : dict, optional
        Mapping for substitution *after* the main operation.
    """
    PHYSICS_LOGGER.info("%s → %s", operation, expression)

    try:
        syms = symbols(variables) if variables else ()
        if not isinstance(syms, (list, tuple)):
            syms = (syms,)
        local_dict = {s.name: s for s in syms}
        expr = sympify(expression, locals=local_dict)

        reasoning = ""
        result: Any

        match operation:
            case "solve":
                if not solve_for:
                    return _prepare_return_dict("error", reason="'solve_for' required")
                targets = symbols(solve_for)
                if not isinstance(targets, (list, tuple)):
                    targets = (targets,)
                result = solve(expr, *targets)
                reasoning = f"Solved for {solve_for}."

            case "diff":
                if not wrt:
                    return _prepare_return_dict("error", reason="'wrt' required for diff")
                result = diff(expr, symbols(wrt))
                reasoning = f"Differentiated w.r.t {wrt}."

            case "integrate":
                if not wrt:
                    return _prepare_return_dict("error", reason="'wrt' required for integrate")
                result = integrate(expr, symbols(wrt))
                reasoning = f"Integrated w.r.t {wrt}."

            case "simplify":
                result = expr.simplify()
                reasoning = "Simplified the expression."

            case "assign":
                if len(syms) != 1:
                    return _prepare_return_dict("error", reason="'assign' needs exactly one variable")
                val = sympify(expression)
                reasoning = f"Assigned {val} to {syms[0].name}."
                return _prepare_return_dict("success", result={syms[0].name: float(val)}, reasoning=reasoning)

            case "subs":
                if not at_point:
                    return _prepare_return_dict("error", reason="'at_point' required for subs")
                result = expr.subs(at_point)
                reasoning = f"Substituted {at_point}."

            case _:
                return _prepare_return_dict("error", reason=f"Unknown operation: {operation}")

        # Second‑stage substitution if requested
        if at_point and operation != "subs":
            substituted = result.subs(at_point) if hasattr(result, "subs") else result
            reasoning += f" Substituted {at_point}."
            return _prepare_return_dict(
                "success",
                result=float(substituted) if substituted.is_number else substituted,
                latex_repr=latex(substituted),
                reasoning=reasoning,
            )

        return _prepare_return_dict("success", result=result, latex_repr=latex(result), reasoning=reasoning)

    except (SympifyError, ValueError, TypeError) as exc:
        msg = f"Symbolic manipulation failed: {exc}"
        PHYSICS_LOGGER.error(msg, exc_info=True)
        return _prepare_return_dict("error", reason=msg)

# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def vector_operation(
    operation: str,
    vectors: List[List[float]],
    *,
    scalar: Optional[float] = None,
) -> Dict[str, Any]:
    """Perform basic vector algebra.

    operation ∈ {``dot``, ``cross``, ``scalar_mult``, ``magnitude``, ``normalize``}.
    """
    PHYSICS_LOGGER.info("Vector op %s on %s", operation, vectors)

    try:
        mats = [Matrix(v) for v in vectors]
        reasoning = ""
        result: Union[Matrix, float]

        match operation:
            case "dot":
                if len(mats) != 2:
                    return _prepare_return_dict("error", reason="Dot product needs two vectors")
                result = mats[0].dot(mats[1])
                reasoning = "Computed dot product."

            case "cross":
                if len(mats) != 2:
                    return _prepare_return_dict("error", reason="Cross product needs two vectors")
                if any(m.shape != (3, 1) for m in mats):
                    return _prepare_return_dict("error", reason="Cross product only for 3D vectors")
                result = mats[0].cross(mats[1])
                reasoning = "Computed cross product."

            case "scalar_mult":
                if len(mats) != 1 or scalar is None:
                    return _prepare_return_dict("error", reason="Provide one vector + scalar")
                result = mats[0] * scalar
                reasoning = f"Scaled vector by {scalar}."

            case "magnitude":
                if len(mats) != 1:
                    return _prepare_return_dict("error", reason="Magnitude needs one vector")
                result = mats[0].norm()
                reasoning = "Computed magnitude."

            case "normalize":
                if len(mats) != 1:
                    return _prepare_return_dict("error", reason="Normalization needs one vector")
                result = mats[0].normalized()
                reasoning = "Normalized vector."

            case _:
                return _prepare_return_dict("error", reason=f"Unknown op: {operation}")

        return _prepare_return_dict("success", result=result, latex_repr=latex(result), reasoning=reasoning)

    except Exception as exc:  # pylint: disable=broad-except
        msg = f"Vector operation failed: {exc}"
        PHYSICS_LOGGER.error(msg, exc_info=True)
        return _prepare_return_dict("error", reason=msg)
