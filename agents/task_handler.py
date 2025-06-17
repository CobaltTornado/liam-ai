import re
import json
import logging
import base64
import asyncio
from PIL import Image
import io
from typing import Optional, List, Dict, Any
import google.generativeai.protos as protos
from agents.base_handler import BaseModeHandler
from tools.git_tools import git_status
from tools.file_system_tools import get_fs_state_str
from tools.prompt_templates import get_standard_planning_prompt, get_deep_reasoning_prompt, get_self_correction_prompt

# Forward declaration for type hinting
if False:
    from main_agent import TaskExecutionContext, ChiefArchitectAgent

AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")
MAX_RETRIES = 2 # Set a limit for self-correction attempts

class TaskModeHandler(BaseModeHandler):
    """
    Handles complex, multi-step tasks by orchestrating planning, execution,
    and a new self-correction loop.
    """

    async def handle(self, prompt: str, image_data: Optional[str] = None,
                     task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):
        if not task_context:
            raise ValueError("TaskModeHandler requires a TaskExecutionContext to manage state.")

        try:
            # Initial Planning
            plan = await self._perform_planning(task_context, deep_reasoning, image_data)
            if not plan:
                await self.progress_manager.broadcast("final_result", "The agent could not generate a valid plan.")
                return

            await task_context.set_plan(plan)

            # Execution with Self-Correction Loop
            execution_successful, step_outputs = await self._perform_execution_with_correction(task_context)

            # Final Reporting
            await self._generate_final_response(task_context, execution_successful, step_outputs)

        except Exception as e:
            AGENT_LOGGER.error(f"A critical error occurred during task execution: {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"CRITICAL ERROR: The task loop failed. Reason: {e}")
            await self.progress_manager.broadcast("final_result",
                                                  f"The agent failed to complete the task due to a critical error: {e}")

    async def _perform_planning(self, task_context: 'TaskExecutionContext', deep_reasoning: bool,
                                image_data: Optional[str] = None, is_correction: bool = False, error_message: str = "") -> Optional[List[Dict]]:
        if is_correction:
            await self.progress_manager.broadcast("log", "Phase 1.5: Re-planning for Self-Correction...")
            prompt_template = get_self_correction_prompt
            planning_prompt = prompt_template(
                original_prompt=task_context.original_prompt,
                failed_step_id=next((s['id'] for s in task_context.plan if s.get('status') == 'failed'), 'Unknown'),
                current_plan=task_context.plan,
                scratchpad=task_context.scratchpad,
                project_state=get_fs_state_str(),
                error_message=error_message
            )
        else:
            await self.progress_manager.broadcast("log",
                                                  f"Phase 1: Planning (Mode: {'Deep Reasoning' if deep_reasoning else 'Standard'})...")
            prompt_template = get_deep_reasoning_prompt if deep_reasoning else get_standard_planning_prompt
            planning_prompt = prompt_template(task_context.original_prompt, get_fs_state_str())

        planning_content = [planning_prompt]
        if image_data:
            # ... (image handling code remains the same)
            pass

        response_stream = await self.agent.planner_model.generate_content_async(planning_content, stream=True)
        # ... (response streaming and JSON extraction code remains the same)
        full_response_text = ""
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    text_part = chunk.parts[0].text
                    full_response_text += text_part
                    await self.progress_manager.broadcast("reasoning_chunk", text_part)
            except (ValueError, IndexError):
                pass

        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', full_response_text)
        json_str = json_match.group(1) if json_match else full_response_text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            AGENT_LOGGER.error(f"Failed to extract JSON plan from response: {full_response_text}. Error: {e}")
            # In case of correction, we might not want to raise an error but return None
            if is_correction:
                return None
            raise ValueError("The planning model did not return a valid JSON plan.")

    async def _perform_execution_with_correction(self, task_context: 'TaskExecutionContext') -> (bool, Dict):
        """ New main execution loop that incorporates self-correction. """
        step_outputs: Dict[str, Any] = {}
        all_steps_succeeded = True
        current_step_index = 0

        while current_step_index < len(task_context.plan):
            step = task_context.plan[current_step_index]
            step_id = step.get('id', str(current_step_index + 1))

            if step.get('status') == 'completed':
                current_step_index += 1
                continue

            await task_context.update_step_status(step_id, 'in_progress')

            try:
                # Execute the current step
                task_string = step.get('tool_code') or step.get('task')
                if not task_string:
                    raise ValueError("Step is missing a 'task' or 'tool_code' field.")

                tool_name, args, return_key = self._parse_task_string(task_string)
                substituted_args = self._substitute_variables(args, step_outputs)

                await task_context.add_to_scratchpad('TOOL_REQUEST',
                                                     f"Calling {tool_name} with args {substituted_args}")

                if tool_name not in self.agent.tools:
                    raise ValueError(f"Tool '{tool_name}' not found in agent's tool list.")

                tool_function = self.agent.tools[tool_name]
                tool_result = await tool_function(**substituted_args) if asyncio.iscoroutinefunction(
                    tool_function) else tool_function(**substituted_args)

                await task_context.add_to_summary(tool_name, substituted_args, tool_result)

                if tool_result.get("status") == "success":
                    if return_key:
                        result_value = tool_result.get("result")
                        step_outputs[return_key] = result_value
                        await task_context.add_to_scratchpad("STATE_UPDATE",
                                                             f"Stored result in '{return_key}': {result_value}")
                    await task_context.update_step_status(step_id, "completed", f"Result: {tool_result.get('result')}")

                    # --- DYNAMIC REPLANNING (User Request #1) ---
                    # After a successful step, we can re-evaluate the rest of the plan
                    new_plan = await self._reevaluate_plan(task_context, current_step_index, step_outputs)
                    if new_plan:
                        await task_context.add_to_scratchpad("REPLAN", "Re-evaluating and updating the plan after successful step.")
                        task_context.plan = new_plan
                        await self.progress_manager.broadcast("plan", task_context.plan)
                        # The loop will continue with the new plan

                    current_step_index += 1 # Move to the next step

                else: # Tool returned a failure status
                    raise Exception(f"Tool '{tool_name}' failed: {tool_result.get('reason', 'Unknown error')}")

            except Exception as e:
                AGENT_LOGGER.error(f"Failed to execute step {step_id}: {e}", exc_info=True)
                await task_context.update_step_status(step_id, "failed", str(e))
                all_steps_succeeded = False

                # --- SELF-CORRECTION (User Request #2) ---
                if task_context.retries < MAX_RETRIES:
                    task_context.retries += 1
                    await task_context.add_to_scratchpad("SELF_CORRECTION", f"Attempting self-correction, retry #{task_context.retries}.")

                    corrected_plan = await self._perform_planning(task_context, deep_reasoning=True, is_correction=True, error_message=str(e))

                    if corrected_plan:
                        await task_context.add_to_scratchpad("REPLAN_SUCCESS", "Successfully generated a new plan.")
                        task_context.plan = corrected_plan
                        await self.progress_manager.broadcast("plan", task_context.plan)
                        current_step_index = 0  # Restart execution from the beginning of the new plan
                        all_steps_succeeded = True # Reset success flag for the new attempt
                        continue # Restart the while loop
                    else:
                        await task_context.add_to_scratchpad("REPLAN_FAILED", "Failed to generate a corrected plan.")
                        break # Exit loop if correction fails
                else:
                    await task_context.add_to_scratchpad("MAX_RETRIES_REACHED", "Maximum self-correction retries reached. Aborting task.")
                    break # Exit loop if max retries are reached

        return all_steps_succeeded, step_outputs


    async def _reevaluate_plan(self, task_context: 'TaskExecutionContext', current_step_index: int, step_outputs: Dict) -> Optional[List[Dict]]:
        """
        A lighter-weight check to see if the plan needs to be adjusted after a successful step.
        For now, this is a placeholder for a more complex implementation.
        A simple check could be to see if a critical file was deleted or a key variable's value is unexpected.
        Returning 'None' means the plan should continue as is.
        """
        # This is where more complex logic could go. For example, you could call the planner
        # with a prompt like "Given the last action's result, is the rest of the plan still optimal?"
        # For now, we will not re-plan after every success to avoid excessive LLM calls,
        # focusing on the more critical self-correction on failure.
        return None

    def _parse_task_string(self, task_string: str) -> (str, Dict[str, Any], Optional[str]):
        """
        Parses the agent's task string to extract the tool, arguments, and an optional return key.
        Example: "solve_expression(expression='(v_initial + 10)', return='final_v')"
        """
        match = re.match(r'([\w_]+)\s*\((.*)\)', task_string)
        if not match:
            # Fallback for simple tool names without arguments. If the string
            # is plain English, treat it as a no-op log message so execution
            # can continue without failure.
            if re.match(r'^[\w_]+$', task_string):
                return task_string, {}, None
            return "log_message", {"message": task_string}, None

        tool_name = match.group(1)
        args_str = match.group(2).strip()
        if not args_str:
            return tool_name, {}, None

        args = {}
        return_key = None

        # This regex handles key='value' or key="value" pairs, including the
        # optional 'return' key. It also supports nested parentheses in the
        # value string.
        pattern = re.compile(r"([\w_]+)\s*=\s*(?:'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\")")
        # It's important to process the string from left to right
        last_pos = 0
        for match_obj in pattern.finditer(args_str):
            key = match_obj.group(1)
            value = match_obj.group(2) if match_obj.group(2) is not None else match_obj.group(3)
            # This check ensures we don't misinterpret parts of a string value as another argument
            if match_obj.start() < last_pos:
                continue
            last_pos = match_obj.end()

            # Un-escape quotes if they were doubled inside the string
            value = value.replace("''", "'").replace('""', '"')
            if key == 'return':
                return_key = value
            else:
                args[key] = value

        return tool_name, args, return_key


    def _substitute_variables(self, args: Dict[str, str], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitutes variable placeholders in argument values with their stored values from the state dictionary.
        """
        substituted_args = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Regex to find words that could be variables
                potential_vars = re.findall(r'\b([a-zA-Z_][\w]*)\b', value)
                for var in potential_vars:
                    if var in outputs:
                        # Replace the found variable with its value from memory
                        # Ensure the replacement is done as a whole word
                        replacement_val = str(outputs[var])
                        # If the value in memory is a list or dict, JSON stringify it
                        if isinstance(outputs[var], (list, dict)):
                           replacement_val = json.dumps(outputs[var])

                        value = re.sub(r'\b' + re.escape(var) + r'\b', replacement_val, value)
            substituted_args[key] = value
        return substituted_args

    async def _generate_final_response(self, task_context: 'TaskExecutionContext', execution_successful: bool,
                                       step_outputs: Dict[str, Any]):
        await self.progress_manager.broadcast("log", "Generating final response...")

        final_status_summary = "All steps were completed successfully."
        if not execution_successful:
            failed_steps = [s.get('id', 'N/A') for s in task_context.plan if s.get('status') == 'failed']
            step_id = failed_steps[0] if failed_steps else 'the last'
            final_status_summary = f"The process stopped because step {step_id} failed, even after {task_context.retries} correction attempt(s)."


        # --- NEW LATEX AGGREGATION LOGIC ---
        latex_steps = []
        if execution_successful:
            for i, step in enumerate(task_context.plan):
                # Ensure we don't go out of bounds if execution stopped early
                if i >= len(task_context.execution_summary):
                    break
                summary_entry = task_context.execution_summary[i]
                tool_result = summary_entry.get("result", {})
                if tool_result.get("status") == "success" and "latex_representation" in tool_result:
                    reasoning = step.get("reasoning", "Calculate next step")
                    latex_formula = tool_result.get("latex_representation")
                    # Format as a LaTeX comment and a formula for display
                    latex_steps.append(f"% {i + 1}. {reasoning}\n\\rightarrow {latex_formula}")

        if latex_steps:
            # Join all LaTeX steps with a double backslash for new lines in display mode
            full_latex_breakdown = "\\\\\n".join(latex_steps)

            # Try to find the name of the last variable that was returned
            final_answer_key = next((self._parse_task_string(p.get('task'))[2] for p in reversed(task_context.plan) if self._parse_task_string(p.get('task'))[2]), None)
            final_answer_val = step_outputs.get(final_answer_key, 'See breakdown')

            # Format the final answer to a reasonable number of decimal places if it's a float
            try:
                final_answer = f"{float(final_answer_val):.2f}"
            except (ValueError, TypeError):
                final_answer = str(final_answer_val)

            await self.progress_manager.broadcast("chat_chunk",
                                                  "I've solved the problem step-by-step. Here is the mathematical breakdown:")
            await self.progress_manager.broadcast("latex_canvas", full_latex_breakdown)
            await self.progress_manager.broadcast("chat_chunk",
                                                  f"\nBased on the steps above, the final result for **{final_answer_key}** is: **{final_answer}**")
        # --- END NEW LOGIC ---
        else:
            # Fallback to original summary logic if no LaTeX was generated
            execution_summary = task_context.get_formatted_summary()
            summary_prompt = (
                f"You just attempted to solve this problem: '{task_context.original_prompt}'.\n\n"
                f"Here is a summary of the actions you took:\n{execution_summary}\n\n"
                f"{final_status_summary}\n\n"
                "Provide a friendly, conversational response summarizing the outcome for the user. If the task was successful, clearly state the final calculated answer(s)."
            )

            response_stream = await self.agent.chat_model.generate_content_async(summary_prompt, stream=True)
            async for chunk in response_stream:
                try:
                    if chunk.parts: await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
                except (ValueError, IndexError):
                    pass

        await self.progress_manager.broadcast("final_result", f"Task finished. {final_status_summary}")
