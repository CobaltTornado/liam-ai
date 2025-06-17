import json
from tools.physics_tools import symbolic_manipulation, vector_operation
from tools.math_tools import solve_expression, calculate_descriptive_statistics, probability_distribution
from tools.file_system_tools import create_file, read_file, update_file, delete_file, list_files
from tools.git_tools import git_commit, git_status


def get_available_tools_docstring():
    """Generates a string containing the docstrings of all available tools."""
    tools = [
        symbolic_manipulation, vector_operation, solve_expression,
        calculate_descriptive_statistics, probability_distribution,
        create_file, read_file, update_file, delete_file, list_files,
        git_commit, git_status
    ]
    docstrings = []
    for tool in tools:
        try:
            args = ", ".join(tool.__annotations__.keys())
            doc = tool.__doc__.strip().split('\n')[0]
            docstring = f"- {tool.__name__}({args}): {doc}"
            docstrings.append(docstring)
        except Exception:
            continue
    return "\n".join(docstrings)


def get_standard_planning_prompt(user_prompt: str, project_state: str) -> str:
    """Returns the standard planning prompt with enhanced logic and a tool menu."""
    tool_docs = get_available_tools_docstring()
    return f"""
    You are an expert AI project architect. Your task is to create a JSON array of numbered steps to fulfill the user's request.

    **AVAILABLE TOOLS:**
    ```
    {tool_docs}
    ```

    **CRITICAL PLANNING INSTRUCTIONS:**
    1.  **Analyze Request:** Determine if the request is a math/physics problem or a coding/file system task.
    2.  **Use Tools Directly:** Based on the AVAILABLE TOOLS, create a plan that calls the tools with the exact arguments specified in their docstrings.
    3.  **Prioritize Specialized Tools:** For physics or math problems, use `symbolic_manipulation` or `vector_operation`. Do not write code to a file unless absolutely necessary.

    User Request: "{user_prompt}"
    Current Project State: {project_state}

    Generate the most direct and efficient JSON plan now, using only the tools listed above.
    """


def get_deep_reasoning_prompt(user_prompt: str, project_state: str) -> str:
    """
    Returns the deep reasoning prompt, now with very explicit instructions on tool selection.
    """
    tool_docs = get_available_tools_docstring()
    return f"""
    You are an expert AI planner. Your task is to create a JSON plan to solve the user's request using ONLY the tools provided.

    **AVAILABLE TOOLS:**
    ```
    {tool_docs}
    ```

    **CRITICAL PLANNING INSTRUCTIONS:**
    1.  **Analyze the Request:** The user wants to solve a multi-step numerical physics problem.
    2.  **CHOOSE THE RIGHT TOOL:**
        * For ALL numerical calculations, including trigonometry (sin, cos, tan) and using constants like pi, you MUST use the `solve_expression` tool.
        * DO NOT use `symbolic_manipulation` for numerical evaluation. Use it ONLY for symbolic algebra (e.g., solving 'x+y=z' for 'x').
        * This ENTIRE physics problem can be solved using a sequence of calls to `solve_expression`.
    3.  **STATE MANAGEMENT:**
        * To save a result, add `return='variable_name'` to your `solve_expression` call. (e.g., `..., return='Fx')`).
        * To use a saved result, place the variable name directly in the next expression. (e.g., `expression='Fx - 10'`).
    4.  **JSON FORMAT:**
        * The output MUST be a JSON array of objects.
        * Each object MUST have these keys: "id", "task", "status", "reasoning".
        * The "task" field MUST be a string containing a direct call to `solve_expression`. For example: `"task": "solve_expression(expression='5*9.8', return='weight')"`.
        * Do not add any extra keys or use descriptive text in the "task" field.

    User Request: "{user_prompt}"
    Current Project State: {project_state}
    Generate the plan using `solve_expression` for all steps.
    """


def get_self_correction_prompt(original_prompt: str, failed_step_id: int, current_plan: list[dict],
                               scratchpad: list[dict], project_state: str) -> str:
    """
    Returns a prompt designed to guide the agent in self-correction after a failed execution step.
    """
    plan_str = json.dumps(current_plan, indent=2)
    scratchpad_str = json.dumps(scratchpad, indent=2)
    tool_docs = get_available_tools_docstring()

    return f"""
    A previous plan failed. Analyze the failure and propose a revised JSON plan.

    **AVAILABLE TOOLS:**
    ```
    {tool_docs}
    ```
    User's Original Request: "{original_prompt}"
    Failed Step ID: {failed_step_id}
    Current Plan: {plan_str}
    Scratchpad: {scratchpad_str}
    Current Project State: {project_state}

    Critically analyze what went wrong. Propose a revised plan starting from step {failed_step_id} using ONLY the AVAILABLE TOOLS and following state management rules.

    Generate the REVISED JSON plan now.
    """
