# agents.py
"""
Agent imports for Agent Studio prototype.
"""

from .base_agent import BaseAgent
from .researcher_agent import ResearcherAgent
from .writer_agent import WriterAgent
from .reviewer_agent import ReviewerAgent
from .editor_agent import EditorAgent
from .seo_agent import SEOAgent
from .fact_checker_agent import FactCheckerAgent
from .planner_agent import PlannerAgent

from langchain.llms import Groq
from crewai import Agent as CrewAgent
from utils import GROQ_API_KEY

class Agent:
    def __init__(self, name, role, description):
        self.name = name
        self.role = role
        self.description = description
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.crew_agent = CrewAgent(name=name, role=role, goal=description, llm=self.llm)

    def run(self, task):
        # The agent processes a task using its LLM
        return self.crew_agent.run(task)

def create_agent(name, role, description):
    """Factory function to create an agent."""
    return Agent(name, role, description) 