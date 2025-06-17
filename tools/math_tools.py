import math
import logging
import numpy as np
import re
from scipy.stats import norm, t, chi2, f
from typing import List, Dict, Union
from sympy import sympify, latex, SympifyError  # Add this import

# --- Logging Setup ---
MATH_LOGGER = logging.getLogger("MathToolV2")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def _create_safe_eval_scope() -> Dict:
    """
    Creates a secured scope for the eval function, including math, numpy,
    and convenience functions for degree-based trigonometry.
    """
    # Start with all functions from the math module, like sin, cos, etc.
    scope = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}

    # Add the entire math module for access to constants like math.pi
    scope['math'] = math

    # Add common numpy functions
    for func_name in ['array', 'linspace', 'logspace', 'mean', 'median', 'std', 'var', 'min', 'max', 'sum', 'prod']:
        if hasattr(np, func_name):
            scope[func_name] = getattr(np, func_name)

    # Add convenience trig helpers that accept degrees directly
    scope['sin_deg'] = lambda x: math.sin(math.radians(x))
    scope['cos_deg'] = lambda x: math.cos(math.radians(x))
    scope['tan_deg'] = lambda x: math.tan(math.radians(x))

    # Add safe built-ins
    scope['abs'] = abs
    scope['round'] = round
    scope['len'] = len
    return scope

def _prepare_return_dict(status: str, result: any = None, reason: str = None, latex: str = None) -> Dict:
    """Formats the standard return dictionary for all tool functions."""
    response = {"status": status}
    if result is not None:
        response["result"] = result
    if reason:
        response["reason"] = reason
    if latex:
        response["latex_representation"] = latex
    return response

# --- Auto-conversion Helper ---
def _convert_trig_degrees(expr: str) -> str:
    """Convert trig functions with degree arguments to radians automatically."""
    pattern = re.compile(r"(sin|cos|tan)\(([^()]+)\)")

    def repl(match: re.Match) -> str:
        func, arg = match.group(1), match.group(2).strip()
        # Check for numeric argument
        try:
            val = float(arg)
            if abs(val) > 2 * math.pi:
                return f"{func}(radians({arg}))"
        except Exception:
            if re.search(r"deg|degree", arg, re.IGNORECASE):
                return f"{func}(radians({arg}))"
        return f"{func}({arg})"

    return pattern.sub(repl, expr)

# --- Core Tool Functions ---

def solve_expression(expression: str) -> Dict:
    """
    Safely evaluates a string-based mathematical expression and provides its LaTeX form.
    Trigonometric functions with degree arguments are automatically converted to radians.

    Args:
        expression (str): The mathematical expression to compute.
                          e.g., "np.mean([1, 2, 3]) * math.pi"

    Returns:
        Dict: A dictionary with the evaluation result, LaTeX representation, or an error.
    """
    MATH_LOGGER.info(f"Evaluating expression: {expression}")
    expression = _convert_trig_degrees(expression)
    safe_scope = _create_safe_eval_scope()
    latex_str = None
    try:
        # First, generate LaTeX representation for display
        try:
            # For cleaner LaTeX, we can sympify the expression.
            # This is for display only, the eval uses the original string.
            display_expr_str = expression.replace('np.', '').replace('math.', '')
            sympy_expr = sympify(display_expr_str)
            latex_str = latex(sympy_expr)
        except (SympifyError, TypeError, Exception):
            # If LaTeX generation fails, fall back to the raw expression.
            latex_str = expression

        # Second, evaluate the expression to get the numerical result
        result = eval(expression, {"__builtins__": {}}, safe_scope)

        if isinstance(result, np.ndarray):
            result = result.tolist()
        MATH_LOGGER.info(f"Expression result: {result}")
        # Include the generated LaTeX in the successful response
        return _prepare_return_dict("success", result=result, latex=latex_str)
    except Exception as e:
        error_message = f"Failed to evaluate expression '{expression}': {e}"
        MATH_LOGGER.error(error_message)
        return _prepare_return_dict("error", reason=error_message)

def probability_distribution(dist_type: str, x: float, **params) -> Dict:
    """
    Calculates the PDF/PMF, CDF, or quantile for a given probability distribution.

    Args:
        dist_type (str): The type of distribution ('normal', 't', 'chi2', 'f').
        x (float): The value at which to evaluate the function (or probability for quantile).
        **params: Distribution-specific parameters.
                  - For 'normal': mu (mean), sigma (std dev).
                  - For 't': df (degrees of freedom).
                  - For 'chi2': df (degrees of freedom).
                  - For 'f': dfn (numerator df), dfd (denominator df).
                  - operation (str): 'pdf' (probability density), 'cdf' (cumulative density), 'quantile' (inverse cdf).

    Returns:
        Dict: A dictionary with the calculated probability/value.
    """
    operation = params.pop('operation', 'pdf')
    MATH_LOGGER.info(f"Calculating {operation} for {dist_type} distribution at x={x} with params {params}")
    try:
        if dist_type == 'normal':
            dist = norm(loc=params.get('mu', 0), scale=params.get('sigma', 1))
        elif dist_type == 't':
            dist = t(df=params['df'])
        elif dist_type == 'chi2':
            dist = chi2(df=params['df'])
        elif dist_type == 'f':
            dist = f(dfn=params['dfn'], dfd=params['dfd'])
        else:
            return _prepare_return_dict("error", reason=f"Unsupported distribution type: {dist_type}")

        if operation == 'pdf':
            result = dist.pdf(x)
        elif operation == 'cdf':
            result = dist.cdf(x)
        elif operation == 'quantile':
            result = dist.ppf(x) # x is the probability for quantile
        else:
            return _prepare_return_dict("error", reason=f"Invalid operation: {operation}")

        return _prepare_return_dict("success", result=result)
    except Exception as e:
        error_message = f"Probability calculation failed: {e}"
        MATH_LOGGER.error(error_message)
        return _prepare_return_dict("error", reason=error_message)

def calculate_descriptive_statistics(data: List[Union[int, float]]) -> Dict:
    """
    Calculates a suite of descriptive statistics for a given dataset.

    Args:
        data (List[Union[int, float]]): A list of numerical data points.

    Returns:
        Dict: A dictionary containing key descriptive statistics.
    """
    if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
        return _prepare_return_dict("error", reason="Input must be a list of numbers.")
    if not data:
        return _prepare_return_dict("error", reason="Input data list cannot be empty.")

    MATH_LOGGER.info(f"Calculating descriptive statistics for a dataset of size {len(data)}")
    try:
        np_data = np.array(data)
        stats = {
            "count": len(np_data),
            "mean": np.mean(np_data),
            "median": np.median(np_data),
            "std_dev": np.std(np_data),
            "variance": np.var(np_data),
            "min": np.min(np_data),
            "max": np.max(np_data),
            "sum": np.sum(np_data),
            "25th_percentile": np.percentile(np_data, 25),
            "75th_percentile": np.percentile(np_data, 75)
        }
        return _prepare_return_dict("success", result=stats)
    except Exception as e:
        error_message = f"Failed to calculate statistics: {e}"
        MATH_LOGGER.error(error_message)
        return _prepare_return_dict("error", reason=error_message)
