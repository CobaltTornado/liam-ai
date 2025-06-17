import logging
import json
import base64
from PIL import Image
import io
from typing import Optional, Dict
from agents.base_handler import BaseModeHandler
from agents.task_handler import TaskModeHandler  # Import the TaskModeHandler to reuse its execution logic

# Forward declaration for type hinting
if False:
    from main_agent import TaskExecutionContext, ChiefArchitectAgent

AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")


class PhysicsModeHandler(BaseModeHandler):
    """
    Handles physics problems by first transcribing them from text or images,
    assessing their difficulty to decide on a reasoning strategy,
    then creating and executing a step-by-step plan using tools.
    """

    async def handle(self, prompt: str, image_data: Optional[str] = None,
                     task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):  # The initial deep_reasoning flag can serve as an override
        if not task_context:
            raise ValueError("PhysicsModeHandler requires a TaskExecutionContext.")

        await self.progress_manager.broadcast("log", "Engaging in Structured Physics Mode...")

        # --- Phase 1: Transcribe the Problem ---
        await self.progress_manager.broadcast("log", "Phase 1: Transcribing the problem from the input...")
        transcribed_problem = await self._transcribe_problem(prompt, image_data)
        if not transcribed_problem:
            await self.progress_manager.broadcast("final_result", "Could not understand the problem from the input.")
            return
        await self.progress_manager.broadcast("log", f"Transcribed Problem: {transcribed_problem}")

        # --- Phase 2: Assess Difficulty to Determine Reasoning Strategy ---
        await self.progress_manager.broadcast("log", "Phase 2: Assessing problem difficulty...")
        difficulty_assessment = await self._assess_problem_difficulty(transcribed_problem)

        # Use deep reasoning if the problem is medium/hard, or if the user explicitly requested it
        use_deep_reasoning = deep_reasoning or (difficulty_assessment.get("difficulty", "easy") in ["medium", "hard"])

        reasoning_mode = "Deep Reasoning" if use_deep_reasoning else "Standard Reasoning"
        await self.progress_manager.broadcast("log",
                                              f"Assessed difficulty: {difficulty_assessment.get('difficulty', 'unknown')}. Justification: {difficulty_assessment.get('justification', 'N/A')}")
        await self.progress_manager.broadcast("log", f"Proceeding with {reasoning_mode}.")

        # --- Create a Task Handler to leverage its logic ---
        task_handler_for_physics = TaskModeHandler(self.agent)
        task_context.original_prompt = transcribed_problem  # Update context with the clean, transcribed problem

        # --- Phase 3 & 4: Plan and Execute ---
        try:
            await task_handler_for_physics.handle(
                prompt=transcribed_problem,
                image_data=None,
                task_context=task_context,
                deep_reasoning=use_deep_reasoning  # Use the dynamically determined flag
            )
        except Exception as e:
            AGENT_LOGGER.error(f"An error occurred during the planning/execution phase of the physics problem: {e}",
                               exc_info=True)
            await self.progress_manager.broadcast("log", f"ERROR: Physics task failed - {e}")
            await self.progress_manager.broadcast("final_result",
                                                  f"The agent failed to solve the physics problem. Error: {e}")

    async def _transcribe_problem(self, prompt: str, image_data: Optional[str] = None) -> Optional[str]:
        """Uses the appropriate model to extract a clear text description of the physics problem."""
        transcription_prompt = "You are an expert at reading physics problems. Transcribe the user's request and any text from the image into a single, clear, text-only problem description. Do not solve it. Only state the problem and what needs to be found. Write equations in a linear text format (e.g., use '**' for exponents, '/' for division)."
        content = [transcription_prompt]
        if prompt: content.append(prompt)

        try:
            # If there is image data, we must use the powerful vision model
            if image_data:
                await self.progress_manager.broadcast("log", "Transcribing with Vision Model...")
                model_to_use = self.agent.vision_model
                image_bytes = base64.b64decode(image_data)
                content.append(Image.open(io.BytesIO(image_bytes)))
            # If it's just text, we can use the faster, lighter model
            else:
                await self.progress_manager.broadcast("log", "Transcribing with Flash-Lite Model...")
                model_to_use = self.agent.flash_lite_model

            response = await model_to_use.generate_content_async(content)
            return response.text
        except Exception as e:
            AGENT_LOGGER.error(f"Could not process input for transcription: {e}")
            return None


    async def _assess_problem_difficulty(self, problem_description: str) -> Dict[str, str]:
        """Uses a light-weight model to rate the difficulty of the transcribed problem."""
        try:
            await self.progress_manager.broadcast("log", "Assessing difficulty with JSON Flash-Lite Model...")
            # Use the lighter, JSON-enabled model for this classification task
            json_assessment_model = self.agent.json_flash_lite_model

            assessment_prompt = (
                "You are a physics professor. Based on the following problem description, assess its difficulty. "
                "Consider the number of concepts involved, the complexity of the equations, and the number of steps required for a solution. "
                "Respond with a JSON object containing two keys: 'difficulty' (string: 'easy', 'medium', or 'hard') and 'justification' (a brief string explaining your rating)."
            )

            response = await json_assessment_model.generate_content_async([assessment_prompt, problem_description])

            # The response.text should be a JSON string, so we parse it
            return json.loads(response.text)

        except Exception as e:
            AGENT_LOGGER.error(f"Failed to assess problem difficulty: {e}")
            # Return a default assessment in case of an error
            return {"difficulty": "easy", "justification": "Could not assess difficulty, defaulting to easy."}
