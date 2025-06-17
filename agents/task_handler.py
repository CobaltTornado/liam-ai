import asyncio
import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PIL import Image  # noqa: F401  # Imported for potential future image handling
import google.generativeai.protos as protos  # noqa: F401

from agents.base_handler import BaseModeHandler
from tools.file_system_tools import get_fs_state_str
from tools.git_tools import git_status  # noqa: F401  # Used elsewhere via dynamic calls
from tools.prompt_templates import (
    get_deep_reasoning_prompt,
    get_self_correction_prompt,
    get_standard_planning_prompt,
)

if TYPE_CHECKING:
    from main_agent import ChiefArchitectAgent, TaskExecutionContext


AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")
MAX_RETRIES = 2  # Maximum self-correction attempts


class TaskModeHandler(BaseModeHandler):
    """Handle complex, multi-step tasks with planning, execution, and self-correction."""

    # --------------------------------------------------
    # Public entry-point
    # --------------------------------------------------
    async def handle(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        task_context: Optional["TaskExecutionContext"] = None,
        deep_reasoning: bool = False,
    ) -> None:
        if not task_context:
            raise ValueError("TaskModeHandler requires a TaskExecutionContext to manage state.")

        try:
            # Phase 1 – Planning
            plan = await self._perform_planning(task_context, deep_reasoning, image_data)
            if not plan:
                await self.progress_manager.broadcast(
                    "final_result", "The agent could not generate a valid plan."
                )
                return

            await task_context.set_plan(plan)

            # Phase 2 – Execution (+ self-correction loop)
            execution_successful, step_outputs = await self._perform_execution_with_correction(
                task_context
            )

            # Phase 3 – Report back to user
            await self._generate_final_response(task_context, execution_successful, step_outputs)

        except Exception as exc:  # pragma: no cover – defensive logging
            AGENT_LOGGER.error("Critical error during task execution", exc_info=True)
            await self.progress_manager.broadcast(
                "log", f"CRITICAL ERROR: The task loop failed. Reason: {exc}"
            )
            await self.progress_manager.broadcast(
                "final_result",
                f"The agent failed to complete the task due to a critical error: {exc}",
            )

    # --------------------------------------------------
    # Phase 1 – Planning helpers
    # --------------------------------------------------
    async def _perform_planning(
        self,
        task_context: "TaskExecutionContext",
        deep_reasoning: bool,
        image_data: Optional[str] = None,
        *,
        is_correction: bool = False,
        error_message: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate a JSON plan (optionally for self-correction)."""

        if is_correction:
            await self.progress_manager.broadcast("log", "Phase 1.5: Re-planning for self-correction…")
            planning_prompt = get_self_correction_prompt(
                original_prompt=task_context.original_prompt,
                failed_step_id=next(
                    (s["id"] for s in task_context.plan if s.get("status") == "failed"),
                    "Unknown",
                ),
                current_plan=task_context.plan,
                scratchpad=task_context.scratchpad,
                project_state=get_fs_state_str(),
                error_message=error_message,
            )
        else:
            mode = "Deep Reasoning" if deep_reasoning else "Standard"
            await self.progress_manager.broadcast("log", f"Phase 1: Planning (Mode: {mode})…")
            prompt_template = (
                get_deep_reasoning_prompt if deep_reasoning else get_standard_planning_prompt
            )
            planning_prompt = prompt_template(task_context.original_prompt, get_fs_state_str())

        planning_content = [planning_prompt]
        if image_data:
            # Placeholder for future image-context support
            pass

        response_stream = await self.agent.planner_model.generate_content_async(
            planning_content, stream=True
        )

        full_response_text = ""
        async for chunk in response_stream:
            if chunk.parts:
                text_part = chunk.parts[0].text
                full_response_text += text_part
                await self.progress_manager.broadcast("reasoning_chunk", text_part)

        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text)
        json_str = json_match.group(1) if json_match else full_response_text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            AGENT_LOGGER.error("Failed to extract JSON plan from response", exc_info=True)
            return None if is_correction else None  # Caller decides what to do

    # --------------------------------------------------
    # Phase 2 – Execution & self-correction
    # --------------------------------------------------
    async def _perform_execution_with_correction(
        self, task_context: "TaskExecutionContext"
    ) -> tuple[bool, Dict[str, Any]]:
        """Execute the plan with optional self-correction iterations."""
        step_outputs: Dict[str, Any] = {}
        all_steps_succeeded = True
        current_step_index = 0

        while current_step_index < len(task_context.plan):
            step = task_context.plan[current_step_index]
            step_id = step.get("id", str(current_step_index + 1))

            if step.get("status") == "completed":
                current_step_index += 1
                continue

            await task_context.update_step_status(step_id, "in_progress")

            try:
                task_string = step.get("tool_code") or step.get("task")
                if not task_string:
                    raise ValueError("Step is missing a 'task' or 'tool_code' field.")

                tool_name, args, return_key = self._parse_task_string(task_string)
                substituted_args = self._substitute_variables(args, step_outputs)

                await task_context.add_to_scratchpad(
                    "TOOL_REQUEST", f"Calling {tool_name} with args {substituted_args}"
                )

                if tool_name not in self.agent.tools:
                    raise ValueError(f"Tool '{tool_name}' not found in agent's tool list.")

                tool_function = self.agent.tools[tool_name]
                tool_result = (
                    await tool_function(**substituted_args)
                    if asyncio.iscoroutinefunction(tool_function)
                    else tool_function(**substituted_args)
                )

                await task_context.add_to_summary(tool_name, substituted_args, tool_result)

                if tool_result.get("status") == "success":
                    if return_key:
                        result_value = tool_result.get("result")
                        step_outputs[return_key] = result_value
                        await task_context.add_to_scratchpad(
                            "STATE_UPDATE", f"Stored result in '{return_key}': {result_value}"
                        )
                    await task_context.update_step_status(
                        step_id, "completed", f"Result: {tool_result.get('result')}"
                    )

                    # Optional dynamic re-planning (placeholder)
                    current_step_index += 1
                else:
                    raise RuntimeError(
                        f"Tool '{tool_name}' failed: {tool_result.get('reason', 'Unknown error')}"
                    )

            except Exception as exc:  # pragma: no cover – runtime failures
                AGENT_LOGGER.error("Failed to execute step", exc_info=True)
                await task_context.update_step_status(step_id, "failed", str(exc))
                all_steps_succeeded = False

                if task_context.retries < MAX_RETRIES:
                    task_context.retries += 1
                    await task_context.add_to_scratchpad(
                        "SELF_CORRECTION", f"Attempting self-correction, retry #{task_context.retries}."
                    )

                    corrected_plan = await self._perform_planning(
                        task_context,
                        deep_reasoning=True,
                        is_correction=True,
                        error_message=str(exc),
                    )

                    if corrected_plan:
                        await task_context.add_to_scratchpad(
                            "REPLAN_SUCCESS", "Successfully generated a new plan."
                        )
                        task_context.plan = corrected_plan
                        await self.progress_manager.broadcast("plan", task_context.plan)
                        current_step_index = 0
                        all_steps_succeeded = True
                        continue  # Restart loop with new plan

                    await task_context.add_to_scratchpad(
                        "REPLAN_FAILED", "Failed to generate a corrected plan."
                    )
                else:
                    await task_context.add_to_scratchpad(
                        "MAX_RETRIES_REACHED", "Maximum self-correction retries reached. Aborting task."
                    )
                break  # Exit main loop

        return all_steps_succeeded, step_outputs

    # --------------------------------------------------
    # Utility helpers
    # --------------------------------------------------
    def _parse_task_string(
        self, task_string: str
    ) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Return (tool_name, args, return_key) parsed from a task string."""

        match = re.match(r"([\w_]+)\s*\((.*)\)", task_string)
        if not match:
            # Simple tool name without arguments OR plain-language string treated as log.
            if re.match(r"^[\w_]+$", task_string):
                return task_string, {}, None
            return "log_message", {"message": task_string}, None

        tool_name = match.group(1)
        args_str = match.group(2).strip()
        if not args_str:
            return tool_name, {}, None

        args: Dict[str, Any] = {}
        return_key: Optional[str] = None

        pattern = re.compile(
            r"([\w_]+)\s*=\s*(?:'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\")"
        )

        last_pos = 0
        for match_obj in pattern.finditer(args_str):
            if match_obj.start() < last_pos:
                continue  # Skip if inside a previous value
            last_pos = match_obj.end()

            key = match_obj.group(1)
            value = match_obj.group(2) if match_obj.group(2) is not None else match_obj.group(3)
            value = value.replace("''", "'").replace('""', '"')

            if key == "return":
                return_key = value
            else:
                args[key] = value

        return tool_name, args, return_key

    def _substitute_variables(self, args: Dict[str, str], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholders in `args` with values from `outputs`."""
        substituted_args: Dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                potential_vars = re.findall(r"\b([a-zA-Z_][\w]*)\b", value)
                for var in potential_vars:
                    if var in outputs:
                        replacement_val: str
                        if isinstance(outputs[var], (list, dict)):
                            replacement_val = json.dumps(outputs[var])
                        else:
                            replacement_val = str(outputs[var])
                        value = re.sub(r"\b" + re.escape(var) + r"\b", replacement_val, value)
            substituted_args[key] = value
        return substituted_args

    # --------------------------------------------------
    # Phase 3 – Final user response
    # --------------------------------------------------
    async def _generate_final_response(
        self,
        task_context: "TaskExecutionContext",
        execution_successful: bool,
        step_outputs: Dict[str, Any],
    ) -> None:
        await self.progress_manager.broadcast("log", "Generating final response…")

        if execution_successful:
            final_status_summary = "All steps were completed successfully."
        else:
            failed_steps = [s.get("id", "N/A") for s in task_context.plan if s.get("status") == "failed"]
            step_id = failed_steps[0] if failed_steps else "the last"
            final_status_summary = (
                f"The process stopped because step {step_id} failed, even after {task_context.retries} correction "
                "attempt(s)."
            )

        # Optional LaTeX aggregation (kept as in original design)
        latex_steps: List[str] = []
        if execution_successful:
            for i, step in enumerate(task_context.plan):
                if i >= len(task_context.execution_summary):
                    break
                summary_entry = task_context.execution_summary[i]
                tool_result = summary_entry.get("result", {})
                if tool_result.get("status") == "success" and "latex_representation" in tool_result:
                    reasoning = step.get("reasoning", "Calculate next step")
                    latex_formula = tool_result["latex_representation"]
                    latex_steps.append(f"% {i + 1}. {reasoning}\n\\rightarrow {latex_formula}")

        if latex_steps:
            full_latex_breakdown = "\\\\\n".join(latex_steps)
            final_answer_key = next(
                (
                    self._parse_task_string(p.get("task", ""))[2]
                    for p in reversed(task_context.plan)
                    if self._parse_task_string(p.get("task", ""))[2]
                ),
                None,
            )
            final_answer_val = step_outputs.get(final_answer_key, "See breakdown")
            try:
                final_answer = f"{float(final_answer_val):.2f}"
            except (ValueError, TypeError):
                final_answer = str(final_answer_val)

            await self.progress_manager.broadcast(
                "chat_chunk", "I've solved the problem step-by-step. Here is the mathematical breakdown:"
            )
            await self.progress_manager.broadcast("latex_canvas", full_latex_breakdown)
            await self.progress_manager.broadcast(
                "chat_chunk",
                f"\nBased on the steps above, the final result for **{final_answer_key}** is: **{final_answer}**",
            )
        else:
            execution_summary = task_context.get_formatted_summary()
            summary_prompt = (
                f"You just attempted to solve this problem: '{task_context.original_prompt}'.\n\n"
                f"Here is a summary of the actions you took:\n{execution_summary}\n\n"
                f"{final_status_summary}\n\n"
                "Provide a friendly, conversational response summarizing the outcome for the user. "
                "If the task was successful, clearly state the final calculated answer(s)."
            )

            response_stream = await self.agent.chat_model.generate_content_async(
                summary_prompt, stream=True
            )
            async for chunk in response_stream:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)

        await self.progress_manager.broadcast("final_result", f"Task finished. {final_status_summary}")
