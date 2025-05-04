"""
Reviewer agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class ReviewerAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Reviewer", description="Responsible for reviewing and providing feedback on content."):
        super().__init__(name, role="Reviewer", description=description, llm=llm, tools=tools) 