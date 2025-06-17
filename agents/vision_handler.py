import logging
import base64
from PIL import Image
import io
from typing import Optional
from agents.base_handler import BaseModeHandler

# Forward declaration for type hinting
if False:
    from main_agent import TaskExecutionContext

AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")

class VisionModeHandler(BaseModeHandler):
    """Handles general vision-related questions."""

    async def handle(self, prompt: str, image_data: Optional[str] = None, task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in Vision Mode...")

        system_instruction = "You are an expert at analyzing images. Your task is to respond to the user's request based on the provided image."
        content = [system_instruction]

        if prompt:
            content.append(prompt)

        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                content.append(img)
                await self.progress_manager.broadcast("log", "Image successfully processed for analysis.")
            except Exception as e:
                AGENT_LOGGER.error(f"Could not process image data for vision mode: {e}")
                await self.progress_manager.broadcast("log", "Error: The attached image could not be processed.")
                return
        else:
            await self.agent.chat_handler.handle(
                "It seems you wanted me to analyze an image, but I didn't receive one. Please try again.")
            return

        response_stream = await self.agent.vision_model.generate_content_async(content, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                pass
        await self.progress_manager.broadcast("final_result", "Vision analysis complete.")
