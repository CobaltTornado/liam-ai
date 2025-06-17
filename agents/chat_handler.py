import asyncio
import logging
from typing import Optional
from agents.base_handler import BaseModeHandler

# Forward declaration for type hinting
if False:
    from main_agent import TaskExecutionContext

AGENT_LOGGER = logging.getLogger("ChiefArchitectAgent")

class ChatModeHandler(BaseModeHandler):
    """Handles simple text-only chat interactions."""

    async def handle(self, prompt: str, image_data: Optional[str] = None, task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):
        await self.progress_manager.broadcast("log", "Engaging in chat mode...")
        chat_prompt = f"You are a helpful AI assistant. A user said: '{prompt}'. Respond in a friendly, direct manner."

        response_stream = await self.agent.chat_model.generate_content_async(chat_prompt, stream=True)
        async for chunk in response_stream:
            try:
                if chunk.parts:
                    await self.progress_manager.broadcast("chat_chunk", chunk.parts[0].text)
            except (ValueError, IndexError):
                AGENT_LOGGER.debug("Received an empty stream chunk.")
                pass
        await self.progress_manager.broadcast("final_result", "Chat interaction complete.")
