"""planning_prompt_utils_cleaned.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility helpers for generating *planning prompts* that instruct an agent
how to solve a user's request by orchestrating a predefined tool‑set.

Highlights of this cleaned version
----------------------------------
• **No behavioural changes** – public API is identical (`get_*_prompt` family).
• **DRY & type‑safe** – removed duplication, added full type hints, and
  leveraged the `inspect` module for robust signature formatting.
• **Clearer doc‑strings** – every function now explains exactly what it does
  and the expectations on callers.
• **Internal helper** `_collect_tool_docs()` centralises the menu creation.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Sequence

from tools.file_system_tools import (
    create_file,
    delete_file,
    list_files,
    read_file,
    update_file,
)
from tools.git_tools import git_commit, git_status
from tools.math_tools import (
    calculate_descriptive_statistics,
    probability_distribution,
    solve_expression,
)
from tools.physics_tools import symbolic_manipulation, vector_operation

__all__ = [
    "get_standard_planning_prompt",
    "get_deep_reasoning_prompt",
    "get_self_correction_prompt",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TOOL_REGISTRY: Sequence[Callable[..., Any]] = [
    symbolic_manipulation,
    vector_operation,
    solve_expression,
    calculate_descriptive_statistics,
    probability_distribution,
    create_file,
    read_file,
    update_file,
    delete_file,
    list_files,
    git_commit,
    git_status,
]


def _collect_tool_docs() -> str:
    """Return a formatted list of tool signatures + brief one‑liner docs."""

    tool_lines: List[str] = []
    for tool in _TOOL_REGISTRY:
        try:
            sig = inspect.signature(tool)
            # Render parameters with annotations when available
            params_repr = ", ".join(
                f"{name}: {param.annotation.__name__ if param.annotation is not inspect._empty else 'Any'}"  # type: ignore[attr-defined]  # noqa: E501
                for name, param in sig.parameters.items()
            )
            doc_first_line = (tool.__doc__ or "No description available.").strip().split("\n")[0]
            tool_lines.append(f"- {tool.__name__}({params_repr}): {doc_first_line}")
        except Exception:  # pragma: no cover – best‑effort formatting
            continue
    return "\n".join(tool_lines)


# ---------------------------------------------------------------------------
# Prompt generators (public API)
# ---------------------------------------------------------------------------

def get_standard_planning_prompt(user_prompt: str, project_state: str) -> str:
    """Craft a *standard* planning prompt with the default tool menu."""

    tool_docs = _collect_tool_docs()
    return f"""
You are an expert AI project architect. Your task is to output a **JSON array** of numbered steps that will satisfy the user's request.

**AVAILABLE TOOLS:**
```
{tool_docs}
```
**CRITICAL PLANNING INSTRUCTIONS:**
1. **Analyse the Request** – decide whether it's a math/physics problem or a coding/file‑system task.
2. **Use Tools Directly** – each step should be a direct call using the exact parameter names expected by the tool.
3. **Prioritise Specialised Tools** – e.g. `symbolic_manipulation`, `vector_operation` for physics; avoid writing files unless necessary.
4. **Beware Trig Units** – `sin`, `cos`, `tan` expect radians. Convert degrees (`x * pi / 180` or `math.radians`).
5. **JSON Format** – every object in the array MUST include: `id`, `task`, `status` = "pending", and `reasoning`.

User Request: "{user_prompt}"
Current Project State:
{project_state}

Generate the most direct and efficient JSON plan now, using only the tools listed above.
"""


def get_deep_reasoning_prompt(user_prompt: str, project_state: str) -> str:
    """Return a *deep‑reasoning* planning prompt with stricter tool rules."""

    tool_docs = _collect_tool_docs()
    return f"""
You are an expert AI planner. Produce a **JSON plan** that solves the user's request **using only the tools provided**.

**AVAILABLE TOOLS:**
```
{tool_docs}
```
**CRITICAL PLANNING INSTRUCTIONS:**
1. **Analyse the Request** – the user seeks a multi‑step numerical physics solution.
2. **Choose the Right Tool:**
   • For *all* numerical maths use `solve_expression`.
   • Reserve `symbolic_manipulation` for algebraic (non‑numeric) work.
   • Ensure trigonometric inputs are in radians (convert if necessary).
3. **State Management:**
   • Persist results by adding `return='var_name'` to the `solve_expression` call.
   • Re‑use stored variables directly in subsequent expressions.
4. **JSON Format:**
   • An array of objects, each with `id`, `task`, `status`, `reasoning`.
   • The `task` must be a literal tool invocation string, *no narrative*.

User Request: "{user_prompt}"
Current Project State:
{project_state}
Generate the plan using `solve_expression` for all steps.
"""


def get_self_correction_prompt(
    *,
    original_prompt: str,
    failed_step_id: int,
    current_plan: List[Dict[str, Any]],
    scratchpad: List[Dict[str, Any]],
    project_state: str,
    error_message: str,
) -> str:
    """Build a self‑correction prompt after a failed execution step."""

    tool_docs = _collect_tool_docs()
    plan_json = json.dumps(current_plan, indent=2)
    scratchpad_json = json.dumps(scratchpad, indent=2)

    return f"""
You are an expert AI architect specialising in **self‑correction**. The previous plan has failed; craft a brand‑new plan that succeeds.

**USER'S ORIGINAL REQUEST:**
"{original_prompt}"

**CONTEXT OF FAILURE:**
• Failed Step ID: {failed_step_id}
• Error Message: "{error_message}"
• Plan that failed:
```json
{plan_json}
```
• Execution scratchpad (state before failure):
```json
{scratchpad_json}
```

**AVAILABLE TOOLS:**
```
{tool_docs}
```

Current Project State:
{project_state}

**CORRECTION INSTRUCTIONS:**
1. **Analyse the Error** – root cause? (typo, wrong tool, bad parameters…)
2. **Formulate a New Strategy** – choose tools/params to avoid repetition of the error.
3. **Generate a *complete* New Plan** – replace the old plan entirely; every step should be `status: "pending"`.
4. **Adhere to JSON Format** – array of objects with `id`, `task`, `status`, `reasoning`.

Generate the corrected JSON plan now.
"""
