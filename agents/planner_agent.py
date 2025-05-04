"""
Planner agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Planner", description="Responsible for planning the workflow and assigning tasks."):
        super().__init__(name, role="Planner", description=description, llm=llm, tools=tools) 