"""math_tool_v2_cleaned.py
--------------------------------
A compact, self‑contained collection of helper utilities for evaluating
string‑based mathematical expressions, working with common probability
distributions, and computing descriptive statistics.

Highlights
~~~~~~~~~~
* Merge‑conflict artefacts removed.
* Single, reusable safe‑eval scope builder.
* Full type hints + tidy imports.
* Public API unchanged: ``solve_expression``, ``probability_distribution``,
  ``calculate_descriptive_statistics``.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Union

import numpy as np
from scipy.stats import chi2, f, norm, t
from sympy import SympifyError, latex, sympify

__all__ = [
    "solve_expression",
    "probability_distribution",
    "calculate_descriptive_statistics",
]

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("MathToolV2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_safe_eval_scope() -> Dict[str, Any]:
    """Return a restricted namespace for ``eval`` containing math/NumPy helpers."""
    scope: Dict[str, Any] = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}

    # Select NumPy helpers to expose
    for name in (
        "array",
        "linspace",
        "logspace",
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max",
        "sum",
        "prod",
        "cos",
        "sin",
        "tan",
        "radians",
        "pi",
    ):
        if hasattr(np, name):
            scope[name] = getattr(np, name)
        elif hasattr(math, name):
            scope[name] = getattr(math, name)

    # Extra helpers (degrees input)
    scope.update(
        {
            "sin_deg": lambda x: math.sin(math.radians(x)),
            "cos_deg": lambda x: math.cos(math.radians(x)),
            "tan_deg": lambda x: math.tan(math.radians(x)),
        }
    )

    scope.update({"abs": abs, "round": round, "len": len})  # whitelisted built‑ins
    return scope


def _prepare_return_dict(
    status: str,
    *,
    result: Any | None = None,
    reason: str | None = None,
    latex_repr: str | None = None,
) -> Dict[str, Any]:
    """Standardised response payload for all public functions."""
    payload: Dict[str, Any] = {"status": status}
    if result is not None:
        payload["result"] = result
    if reason:
        payload["reason"] = reason
    if latex_repr:
        payload["latex_representation"] = latex_repr
    return payload


_TRIG_PATTERN = re.compile(r"(sin|cos|tan)\(([^()]+)\)")


def _convert_trig_degrees(expr: str) -> str:
    """Convert *numeric* trig calls given in degrees to radian form."""

    def _repl(match: re.Match[str]) -> str:
        func, arg = match.group(1), match.group(2).strip()
        try:
            val = float(arg)
            if abs(val) > 2 * math.pi:  # heuristic
                return f"{func}(radians({arg}))"
        except ValueError:  # non‑numeric arg
            if re.search(r"deg|degree", arg, re.IGNORECASE):
                return f"{func}(radians({arg}))"
        return f"{func}({arg})"

    return _TRIG_PATTERN.sub(_repl, expr)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_expression(expression: str) -> Dict[str, Any]:
    """Safely evaluate *expression* and return numeric + LaTeX representations."""
    LOGGER.info("Evaluating expression: %s", expression)
    prepared_expr = _convert_trig_degrees(expression).replace("^", "**")
    safe_scope = _create_safe_eval_scope()

    # Build LaTeX (best‑effort)
    try:
        display_expr = prepared_expr.replace("np.", "").replace("math.", "")
        latex_repr = latex(sympify(display_expr))
    except (SympifyError, TypeError):
        latex_repr = prepared_expr  # fallback

    try:
        result = eval(prepared_expr, {"__builtins__": {}}, safe_scope)  # nosec B307
        if isinstance(result, np.ndarray):
            result = result.tolist()
        return _prepare_return_dict("success", result=result, latex_repr=latex_repr)
    except Exception as exc:  # pylint: disable=broad-except
        msg = f"Failed to evaluate expression '{expression}': {exc}"
        LOGGER.error(msg)
        return _prepare_return_dict("error", reason=msg)


# ---------------------------------------------------------------------------
# Probability distributions
# ---------------------------------------------------------------------------

def probability_distribution(
    dist_type: str,
    x: float,
    /,
    **params: Any,
) -> Dict[str, Any]:
    """Compute PDF/PMF, CDF, or quantile for common 1‑D distributions.

    ``dist_type`` ∈ {``'normal'``, ``'t'``, ``'chi2'``, ``'f'``}
    Use ``operation`` kwarg: ``'pdf'`` | ``'cdf'`` | ``'quantile'``.
    """
    operation = params.pop("operation", "pdf")
    LOGGER.info(
        "Calculating %s for %s distribution at x=%s with params %s",
        operation,
        dist_type,
        x,
        params,
    )

    # Choose distribution
    try:
        match dist_type:
            case "normal":
                dist = norm(loc=params.get("mu", 0), scale=params.get("sigma", 1))
            case "t":
                dist = t(df=params["df"])
            case "chi2":
                dist = chi2(df=params["df"])
            case "f":
                dist = f(dfn=params["dfn"], dfd=params["dfd"])
            case _:
                return _prepare_return_dict("error", reason=f"Unsupported distribution type: {dist_type}")

        # Apply operation
        match operation:
            case "pdf":
                result = dist.pdf(x)
            case "cdf":
                result = dist.cdf(x)
            case "quantile":
                result = dist.ppf(x)  # here x is probability
            case _:
                return _prepare_return_dict("error", reason=f"Invalid operation: {operation}")

        return _prepare_return_dict("success", result=result)
    except Exception as exc:  # pylint: disable=broad-except
        msg = f"Probability calculation failed: {exc}"
        LOGGER.error(msg)
        return _prepare_return_dict("error", reason=msg)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def calculate_descriptive_statistics(data: List[Union[int, float]] | np.ndarray) -> Dict[str, Any]:
    """Return common descriptive statistics for *data* (list or ``np.ndarray``)."""
    if not isinstance(data, (list, np.ndarray)) or not all(isinstance(x, (int, float)) for x in data):
        return _prepare_return_dict("error", reason="Input must be a list of numbers.")

    if len(data) == 0:
        return _prepare_return_dict("error", reason="Input data list cannot be empty.")

    LOGGER.info("Calculating descriptive statistics for dataset of length %d", len(data))

    try:
        arr = np.asarray(data, dtype=float)
        stats = {
            "count": arr.size,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr, ddof=0)),
            "variance": float(np.var(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
            "25th_percentile": float(np.percentile(arr, 25)),
            "75th_percentile": float(np.percentile(arr, 75)),
        }
        return _prepare_return_dict("success", result=stats)
    except Exception as exc:  # pylint: disable=broad-except
        msg = f"Failed to calculate statistics: {exc}"
        LOGGER.error(msg)
        return _prepare_return_dict("error", reason=msg)
