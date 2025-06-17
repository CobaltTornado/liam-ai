import os
import logging
import asyncio
import threading
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai.protos as protos
from abc import ABC, abstractmethod
import base64
from PIL import Image
import io
import json
from typing import Optional

# Import the logger class for type hinting
if False:
    from request_logger import RequestLogger

# Tool Imports
from tools.git_tools import git_commit, git_status
from tools.file_system_tools import create_file, read_file, update_file, delete_file, list_files, get_fs_state_str
from tools.math_tools import solve_expression, probability_distribution, calculate_descriptive_statistics
from tools.physics_tools import symbolic_manipulation, vector_operation

# Agent Handler Imports
from agents.base_handler import BaseModeHandler
from agents.chat_handler import ChatModeHandler
from agents.vision_handler import VisionModeHandler
from agents.physics_handler import PhysicsModeHandler
from agents.task_handler import TaskModeHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")


class ProgressManager:
    """Manages WebSocket connections and broadcasts messages, now with logging."""

    def __init__(self, logger: Optional['RequestLogger'] = None):
        self.connections = set()
        self._lock = threading.Lock()
        self.logger = logger
        AGENT_LOGGER.info("Progress Manager initialized.")

    def add_connection(self, ws):
        with self._lock:
            self.connections.add(ws)
        AGENT_LOGGER.info(f"Client connected. Total connections: {len(self.connections)}")

    def remove_connection(self, ws):
        with self._lock:
            self.connections.discard(ws)
        AGENT_LOGGER.info(f"Client disconnected. Total connections: {len(self.connections)}")

    def _broadcast_sync(self, data: dict):
        if self.logger:
            self.logger.log_agent_broadcast(data)

        message = json.dumps(data)
        closed_connections = set()
        with self._lock:
            for ws in list(self.connections):
                try:
                    ws.send(message)
                except Exception as e:
                    AGENT_LOGGER.warning(f"Failed to send to a client, removing: {e}")
                    closed_connections.add(ws)
        if closed_connections:
            with self._lock:
                self.connections.difference_update(closed_connections)

    async def broadcast(self, data_type: str, payload: any):
        data = {"type": data_type, "payload": payload}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._broadcast_sync, data)


class TaskExecutionContext:
    """Holds all state for a single, complex task execution."""

    def __init__(self, original_prompt: str, progress_manager: ProgressManager):
        self.original_prompt = original_prompt
        self.plan: list[dict] = []
        self.scratchpad: list[dict] = []
        self.execution_summary: list[dict] = []
        self.progress_manager = progress_manager
        self.retries = 0  # Add a retry counter for self-correction
        AGENT_LOGGER.info(f"TaskExecutionContext created for prompt: '{original_prompt[:50]}...'")

    async def add_to_scratchpad(self, entry_type: str, detail: str):
        entry = {'type': entry_type, 'detail': detail}
        self.scratchpad.append(entry)
        await self.progress_manager.broadcast("log", f"Scratchpad: {entry_type} - {detail}")

    async def set_plan(self, plan_list: list[dict]):
        self.plan = plan_list
        await self.add_to_scratchpad('INITIAL_PLAN', f'Plan with {len(plan_list)} steps generated.')
        await self.progress_manager.broadcast("plan", self.plan)

    async def update_step_status(self, step_id: int, status: str, detail: str | None = None):
        for step in self.plan:
            if str(step.get('id')) == str(step_id): # Compare as strings for robustness
                step['status'] = status
                if detail: step['detail'] = detail
                await self.add_to_scratchpad('STEP_STATUS_UPDATE', f"Step {step_id} is now {status}.")
                await self.progress_manager.broadcast("plan", self.plan)
                return
        AGENT_LOGGER.warning(f"Attempted to update status for non-existent step_id: {step_id}")

    async def add_to_summary(self, tool_name: str, args: dict, result: dict):
        summary_entry = {"tool": tool_name, "args": args, "result": result}
        self.execution_summary.append(summary_entry)
        await self.progress_manager.broadcast("log", f"Tool Executed: {tool_name} -> {result.get('status', 'OK')}")

    def get_formatted_summary(self) -> str:
        if not self.execution_summary: return "No actions taken yet."
        return "\n".join([f"- {s['tool']}({s['args']}) -> {s['result']}" for s in self.execution_summary])


class ChiefArchitectAgent:
    """The main agent class, now with temperature controls for planning."""

    def __init__(self, progress_manager: ProgressManager, logger: 'RequestLogger'):
        AGENT_LOGGER.info("Initializing ChiefArchitectAgent instance...")
        load_dotenv()
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

        self.logger = logger
        self.progress_manager = progress_manager
        self.progress_manager.logger = logger
        self.logger.log("ChiefArchitectAgent Initialized", "Agent")

        pro_model_name = os.getenv("PRO_MODEL_NAME", "gemini-1.5-pro-latest")
        flash_model_name = os.getenv("FLASH_MODEL_NAME", "gemini-1.5-flash-latest")
        flash_lite_model_name = os.getenv("FLASH_LITE_MODEL_NAME", "gemini-1.0-pro")

        # --- Set low temperature for precision tasks ---
        # A low temperature (e.g., 0.1) makes the model more deterministic and less creative,
        # forcing it to adhere strictly to the prompts and tool definitions.
        low_temp_config = {"temperature": 0.1}

        self.chat_model = genai.GenerativeModel(model_name=flash_model_name)  # Keep chat creative
        self.vision_model = genai.GenerativeModel(model_name=pro_model_name)
        self.flash_lite_model = genai.GenerativeModel(model_name=flash_lite_model_name)

        # Apply low temperature to planning and JSON models
        self.planner_model = genai.GenerativeModel(
            model_name=flash_model_name,
            generation_config=low_temp_config
        )
        self.json_vision_model = genai.GenerativeModel(
            model_name=pro_model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        self.json_flash_lite_model = genai.GenerativeModel(
            model_name=flash_lite_model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )

        self.tools = {
            "create_file": create_file, "read_file": read_file, "update_file": update_file, "delete_file": delete_file,
            "list_files": list_files,
            "git_commit": git_commit, "git_status": git_status,
            "solve_expression": solve_expression,
            "calculate_probability_distribution": probability_distribution,
            "calculate_descriptive_statistics": calculate_descriptive_statistics,
            "symbolic_manipulation": symbolic_manipulation,
            "vector_operation": vector_operation
        }

        self.executor_tool_config = protos.Tool(
            function_declarations=[protos.FunctionDeclaration(name=n, description=f.__doc__) for n, f in
                                   self.tools.items() if f.__doc__])

        # Apply low temperature to the tool executor model as well
        self.executor_model = genai.GenerativeModel(
            model_name=flash_model_name,
            tools=[self.executor_tool_config],
            generation_config=low_temp_config
        )

        self.chat_handler = ChatModeHandler(self)
        self.physics_handler = PhysicsModeHandler(self)
        self.vision_handler = VisionModeHandler(self)
        self.task_handler = TaskModeHandler(self)

        self.current_task_context = None
        self.logger.log("All agent components and handlers initialized.", "Agent")


    async def decide_mode(self, prompt: str, image_data: str | None = None) -> str:
        physics_keywords = ['physics', 'force', 'mass', 'acceleration', 'velocity', 'energy', 'solve', 'calculate',
                            'vector', 'particle', 'projectile', 'kinematics', 'energy']
        task_keywords = ['create', 'build', 'update', 'run', 'execute', 'develop', 'implement', 'make', 'write', 'file',
                         'code', 'project', 'git', 'list', 'delete', 'replan', 'correct']
        vision_keywords = ['image', 'picture', 'see', 'look', 'analyze', 'describe', 'transcribe']
        prompt_lower = prompt.lower() if prompt else ""
        if image_data:
            if any(keyword in prompt_lower for keyword in physics_keywords): return "physics"
            if "solve" in prompt_lower or "calculate" in prompt_lower: return "physics"
            return "vision"
        if any(keyword in prompt_lower for keyword in physics_keywords): return "physics"
        if any(keyword in prompt_lower for keyword in task_keywords): return "task"
        if len(prompt.split()) < 7: return "chat"
        return "task"

    async def execute_task(self, prompt: str, image_data: str | None = None, deep_reasoning: bool = False):
        self.logger.log(f"Executing new task.", "Agent")
        self.current_task_context = None
        final_status = "success"
        final_message = "Task completed successfully."
        try:
            mode = await self.decide_mode(prompt, image_data)
            self.logger.log(f"Agent mode selected: {mode.upper()}", "Agent")
            await self.progress_manager.broadcast("log", f"Agent mode selected: {mode.upper()}")
            handler_map = {"chat": self.chat_handler, "physics": self.physics_handler, "vision": self.vision_handler,
                           "task": self.task_handler}
            if mode in ["task", "physics"]:
                self.current_task_context = TaskExecutionContext(prompt, self.progress_manager)
                await handler_map[mode].handle(prompt, image_data, task_context=self.current_task_context,
                                               deep_reasoning=deep_reasoning)
            else:
                await handler_map[mode].handle(prompt, image_data, deep_reasoning=deep_reasoning)
        except Exception as e:
            final_status = "error"
            final_message = f"The agent failed to complete the task. Error: {e}"
            self.logger.log(f"CRITICAL EXECUTION ERROR: {e}", "Agent")
            AGENT_LOGGER.error(f"Task execution failed for prompt '{prompt}': {e}", exc_info=True)
            await self.progress_manager.broadcast("log", f"ERROR: Task failed - {e}")
            await self.progress_manager.broadcast("final_result", final_message)
        finally:
            self.logger.log(f"Finished task execution.", "Agent")
            self.logger.set_final_summary({"status": final_status, "message": final_message})
