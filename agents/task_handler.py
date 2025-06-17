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


class TaskModeHandler(BaseModeHandler):
    """
    Handles complex, multi-step tasks by orchestrating a planning phase,
    an execution phase with tool use, and a final reporting phase.
    """

    async def handle(self, prompt: str, image_data: Optional[str] = None,
                     task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):
        if not task_context:
            raise ValueError("TaskModeHandler requires a TaskExecutionContext to manage state.")

        try:
            plan = await self._perform_planning(task_context, deep_reasoning, image_data)
            if not plan:
                await self.progress_manager.broadcast("final_result", "The agent could not generate a valid plan.")
                return

            task_context.plan = plan  # Set the plan in the context

            execution_successful, step_outputs = await self._perform_execution(task_context)
            await self._generate_final_response(task_context, execution_successful, step_outputs)

        except Exception as e:
            AGENT_LOGGER.error(f"A critical error occurred during task execution: {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"CRITICAL ERROR: The task loop failed. Reason: {e}")
            await self.progress_manager.broadcast("final_result",
                                                  f"The agent failed to complete the task due to a critical error: {e}")

    async def _perform_planning(self, task_context: 'TaskExecutionContext', deep_reasoning: bool,
                                image_data: Optional[str] = None) -> Optional[List[Dict]]:
        await self.progress_manager.broadcast("log",
                                              f"Phase 1: Planning (Mode: {'Deep Reasoning' if deep_reasoning else 'Standard'})...")
        project_state = f"--- File System ---\n{get_fs_state_str()}"
        prompt_template = get_deep_reasoning_prompt if deep_reasoning else get_standard_planning_prompt

        # We now use the unified prompt template from the updated file
        planning_prompt = prompt_template(task_context.original_prompt, project_state)
        planning_content = [planning_prompt]

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                planning_content.append(Image.open(io.BytesIO(image_bytes)))
                await self.progress_manager.broadcast("log", "Planner is considering the attached image.")
            except Exception as e:
                AGENT_LOGGER.error(f"Could not process image for planner: {e}")

        response_stream = await self.agent.planner_model.generate_content_async(planning_content, stream=True)
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
            raise ValueError("The planning model did not return a valid JSON plan.")

    async def _perform_execution(self, task_context: 'TaskExecutionContext') -> bool:
        await self.progress_manager.broadcast("log", "Phase 2: Execution...")

        # This dictionary is the agent's "memory" for the duration of this task
        step_outputs: Dict[str, Any] = {}
        all_steps_succeeded = True

        for i, step in enumerate(task_context.plan):
            step_id = step.get('id', str(i + 1))
            if step.get('status') == 'completed':
                continue

            await task_context.update_step_status(step_id, 'in_progress')

            try:
                task_string = step['task']
                tool_name, args, return_key = self._parse_task_string(task_string)

                # Substitute variables from memory before calling the tool
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
                else:
                    raise Exception(f"Tool '{tool_name}' failed: {tool_result.get('reason', 'Unknown error')}")

            except Exception as e:
                AGENT_LOGGER.error(f"Failed to execute step {step_id}: {e}", exc_info=True)
                await task_context.update_step_status(step_id, "failed", str(e))
                all_steps_succeeded = False
                break  # Stop execution on first failure

        return all_steps_succeeded, step_outputs

    def _parse_task_string(self, task_string: str) -> (str, Dict[str, Any], Optional[str]):
        """
        Parses the agent's task string to extract the tool, arguments, and an optional return key.
        Example: "solve_expression(expression='(v_initial + 10)', return='final_v')"
        """
        match = re.match(r'([\w_]+)\s*\((.*)\)', task_string)
        if not match:
            raise ValueError(f"Could not parse task string into tool and arguments: {task_string}")

        tool_name = match.group(1)
        args_str = match.group(2).strip()

        args = {}
        return_key = None

        # This regex handles key='value' pairs, including the optional 'return' key
        pattern = re.compile(r"([\w_]+)\s*=\s*'((?:[^']|'')+)'")

        for key, value in pattern.findall(args_str):
            # Un-escape single quotes if they were doubled up inside the string
            value = value.replace("''", "'")
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
                        value = re.sub(r'\b' + re.escape(var) + r'\b', str(outputs[var]), value)
            substituted_args[key] = value
        return substituted_args

    async def _generate_final_response(self, task_context: 'TaskExecutionContext', execution_successful: bool,
                                       step_outputs: Dict[str, Any]):
        await self.progress_manager.broadcast("log", "Generating final response...")

        final_status_summary = "All steps were completed successfully."
        if not execution_successful:
            failed_steps = [s['id'] for s in task_context.plan if s.get('status') == 'failed']
            final_status_summary = f"However, the process stopped because step {failed_steps[0]} failed." if failed_steps else "However, the task failed to complete."

        # Check if a LaTeX breakdown is available from the physics handler
        if execution_successful and 'latex_breakdown' in step_outputs:
            # If we have a latex breakdown, format it nicely for the user
            final_answer = step_outputs.get('acceleration', 'Not found')  # Or whatever the final variable is named

            response_parts = [
                "Great news! I've successfully solved the physics problem for you. Here is the step-by-step breakdown:\n\n",
                "```latex\n",
                step_outputs['latex_breakdown'],
                "\n```\n\n",
                f"The final calculated acceleration of the block is approximately **{final_answer:.2f} m/s²**."
            ]

            for part in response_parts:
                await self.progress_manager.broadcast("chat_chunk", part)
                await asyncio.sleep(0.05)  # Small delay for streaming effect

        else:
            # Fallback to the original summary method if no LaTeX is available or if the task failed
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