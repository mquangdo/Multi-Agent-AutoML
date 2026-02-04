"""
Base Agent class for AutoML System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..config.llm_config import get_llm


class BaseAgent(ABC):
    """Base class for all agents in the AutoML system"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = get_llm()

    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task and return results

        Args:
            task: Dictionary containing task details

        Returns:
            Dictionary containing results
        """
        pass

    def _create_messages(self, system_prompt: str, user_input: str) -> list:
        """Create messages for LLM processing"""
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    def _invoke_llm(self, messages: list) -> str:
        """Invoke LLM with messages"""
        response = self.llm.invoke(messages)
        return response.content
