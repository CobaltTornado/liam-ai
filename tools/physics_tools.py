import logging
from sympy import sympify, solve, diff, integrate, symbols, Matrix, latex, SympifyError
from typing import Dict, List, Optional

# --- Logging Setup ---
PHYSICS_LOGGER = logging.getLogger("PhysicsToolV2")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def _prepare_return_dict(status: str, result: any = None, reason: str = None, latex: str = None, reasoning: str = None) -> Dict:
    """Formats the standard return dictionary for all tool functions."""
    response = {"status": status}
    if result is not None:
        # Handle various result types from SymPy for clean output
        if isinstance(result, list) and result:
            # Check if it's a list of solutions
            response["result"] = [str(item) for item in result]
        elif hasattr(result, 'tolist'): # Handle Matrix
             response["result"] = result.tolist()
        else:
             response["result"] = str(result)
    if reason:
        response["reason"] = reason
    if latex:
        response["latex_representation"] = latex
    if reasoning:
        response["reasoning"] = reasoning
    return response

# --- Core Tool Functions ---

def symbolic_manipulation(
    expression: str,
    operation: str,
    variables: str = "",
    solve_for: Optional[str] = None,
    wrt: Optional[str] = None,
    at_point: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Performs symbolic math operations like solving, differentiation, and integration.

    Args:
        expression (str): The mathematical expression or equation string (e.g., "x**2 + y = z").
        operation (str): The operation to perform. One of ['solve', 'diff', 'integrate', 'simplify', 'subs'].
        variables (str): A comma-separated string of all variables in the expression (e.g., "x,y,z").
        solve_for (str, optional): The variable to solve for if operation is 'solve'.
        wrt (str, optional): "With respect to" variable for differentiation ('diff') or integration ('integrate').
        at_point (Dict[str, float], optional): A dictionary to substitute variables with numerical values after the operation.

    Returns:
        Dict: A dictionary with the result as a string and its LaTeX representation.
    """
    PHYSICS_LOGGER.info(f"Performing '{operation}' on '{expression}'")
    try:
        # Define symbolic variables from the comma-separated string
        syms = symbols(variables) if variables else ()
        if not isinstance(syms, (list, tuple)):
            syms = (syms,)
        expr = sympify(expression, locals={s.name: s for s in syms})
        reasoning_text = ""
        result = None

        if operation == 'solve':
            if not solve_for:
                return _prepare_return_dict("error", reason="'solve_for' is required for 'solve' operation.")
            target_symbol = symbols(solve_for)
            if not isinstance(target_symbol, tuple):
                target_symbol = (target_symbol,)
            solution = solve(expr, *target_symbol)
            result = solution
            reasoning_text = f"Solved the expression for {solve_for}."
        elif operation == 'diff':
            if not wrt:
                return _prepare_return_dict("error", reason="'wrt' is required for 'diff' operation.")
            diff_var = symbols(wrt)
            result = diff(expr, diff_var)
            reasoning_text = f"Differentiated the expression with respect to {wrt}."
        elif operation == 'integrate':
            if not wrt:
                return _prepare_return_dict("error", reason="'wrt' is required for 'integrate' operation.")
            int_var = symbols(wrt)
            result = integrate(expr, int_var)
            reasoning_text = f"Integrated the expression with respect to {wrt}."
        elif operation == 'simplify':
            result = expr.simplify()
            reasoning_text = "Simplified the expression."
        elif operation == 'assign':
            # Simple assignment of a numeric value to a variable
            if len(syms) != 1:
                return _prepare_return_dict("error", reason="'assign' requires exactly one variable")
            value = sympify(expression)
            reasoning_text = f"Assigned the value {value} to {syms[0]}"
            return _prepare_return_dict("success", result={syms[0].name: float(value)}, reasoning=reasoning_text)
        elif operation == 'subs':
             if not at_point:
                return _prepare_return_dict("error", reason="'at_point' is required for 'subs' operation.")
             result = expr.subs(at_point)
             reasoning_text = f"Substituted the variables with the values in {at_point}."
        else:
            return _prepare_return_dict("error", reason=f"Unknown operation: {operation}")

        if at_point and operation != 'subs':
            result_at_point = result.subs(at_point) if hasattr(result, 'subs') else result
            reasoning_text += f" Then, substituted the point {at_point} into the result."
            return _prepare_return_dict("success", result=float(result_at_point), latex=latex(result_at_point), reasoning=reasoning_text)

        return _prepare_return_dict("success", result=result, latex=latex(result), reasoning=reasoning_text)

    except (SympifyError, TypeError, ValueError, Exception) as e:
        error_message = f"Symbolic manipulation failed: {e}"
        PHYSICS_LOGGER.error(error_message, exc_info=True)
        return _prepare_return_dict("error", reason=error_message)

def vector_operation(operation: str, vectors: List[List[float]], scalar: Optional[float] = None) -> Dict:
    """
    Performs vector operations like dot product, cross product, and scalar multiplication.

    Args:
        operation (str): The vector operation ('dot', 'cross', 'scalar_mult', 'magnitude', 'normalize').
        vectors (List[List[float]]): A list containing one or two vectors (e.g., [[1, 2, 3], [4, 5, 6]]).
        scalar (float, optional): The scalar value for scalar multiplication.

    Returns:
        Dict: A dictionary containing the resulting vector or scalar value.
    """
    PHYSICS_LOGGER.info(f"Performing vector '{operation}' on vectors {vectors}")
    try:
        vec_matrices = [Matrix(v) for v in vectors]
        reasoning_text = ""
        result = None

        if operation == 'dot':
            if len(vec_matrices) != 2: return _prepare_return_dict("error", reason="Dot product requires exactly two vectors.")
            result = vec_matrices[0].dot(vec_matrices[1])
            reasoning_text = f"Calculated the dot product of the vectors."
        elif operation == 'cross':
            if len(vec_matrices) != 2: return _prepare_return_dict("error", reason="Cross product requires exactly two vectors.")
            if vec_matrices[0].shape[0] != 3 or vec_matrices[1].shape[0] != 3:
                 return _prepare_return_dict("error", reason="Cross product is only defined for 3D vectors.")
            result = vec_matrices[0].cross(vec_matrices[1])
            reasoning_text = "Calculated the cross product of the 3D vectors."
        elif operation == 'scalar_mult':
            if len(vec_matrices) != 1 or scalar is None:
                return _prepare_return_dict("error", reason="Scalar multiplication requires one vector and one scalar.")
            result = vec_matrices[0] * scalar
            reasoning_text = f"Multiplied the vector by the scalar {scalar}."
        elif operation == 'magnitude':
            if len(vec_matrices) != 1: return _prepare_return_dict("error", reason="Magnitude calculation requires one vector.")
            result = vec_matrices[0].norm()
            reasoning_text = "Calculated the magnitude (norm) of the vector."
        elif operation == 'normalize':
            if len(vec_matrices) != 1: return _prepare_return_dict("error", reason="Normalization requires one vector.")
            result = vec_matrices[0].normalized()
            reasoning_text = "Normalized the vector to unit length."
        else:
            return _prepare_return_dict("error", reason=f"Unknown vector operation: {operation}")

        return _prepare_return_dict("success", result=result, latex=latex(result), reasoning=reasoning_text)

    except Exception as e:
        error_message = f"Vector operation failed: {e}"
        PHYSICS_LOGGER.error(error_message, exc_info=True)
        return _prepare_return_dict("error", reason=error_message)
