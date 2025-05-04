"""
Fact Checker agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class FactCheckerAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Fact Checker", description="Responsible for verifying the accuracy of information."):
        super().__init__(name, role="Fact Checker", description=description, llm=llm, tools=tools) 