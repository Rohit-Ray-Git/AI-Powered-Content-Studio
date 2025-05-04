"""
Writer agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class WriterAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Writer", description="Responsible for writing content based on research."):
        super().__init__(name, role="Writer", description=description, llm=llm, tools=tools) 