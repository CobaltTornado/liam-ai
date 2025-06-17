from abc import ABC, abstractmethod
from typing import Optional

# Forward declaration for type hinting to avoid circular import errors at runtime.
if False:
    from main_agent import ChiefArchitectAgent, TaskExecutionContext


class BaseModeHandler(ABC):
    """Abstract base class for handling different agent modes."""

    def __init__(self, agent_instance: 'ChiefArchitectAgent'):
        """
        Initializes the handler with a reference to the main agent instance.

        Args:
            agent_instance: The instance of the ChiefArchitectAgent.
        """
        self.agent = agent_instance
        self.progress_manager = agent_instance.progress_manager

    @abstractmethod
    async def handle(self, prompt: str, image_data: Optional[str] = None,
                     task_context: Optional['TaskExecutionContext'] = None,
                     deep_reasoning: bool = False):
        """
        The main method to handle a user's request for a specific mode.
        This must be implemented by all subclasses.
        """
        pass
