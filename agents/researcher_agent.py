"""
Researcher agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Researcher", description="Responsible for researching topics and gathering information."):
        super().__init__(name, role="Researcher", description=description, llm=llm, tools=tools) 