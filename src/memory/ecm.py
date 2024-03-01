#This module is from https://github.com/langchain-ai/langchain/issues/1800#issuecomment-1598128244
"""
Extended Conversation Buffer Memory Module.

This class is an extension of the ConversationBufferMemory.
It adds support for extra variables in the conversation buffer memory.
"""

from langchain.memory import ConversationBufferMemory

class ExtendedConversationBufferMemory(ConversationBufferMemory):
    """
    Extended conversation buffer memory class.
    
    This class inherits from ConversationTokenBufferMemory and provides additional functionality to handle extra variables.
    """

    extra_variables: list[str] = []

    @property
    def memory_variables(self) -> list[str]:
        """Will always return a list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: dict[str, any]) -> dict[str, any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d
