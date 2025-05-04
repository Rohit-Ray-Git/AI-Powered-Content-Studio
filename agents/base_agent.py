"""
Base agent class for Agent Studio prototype.
"""

from crewai import Agent as CrewAgent

class BaseAgent:
    def __init__(self, name, role, description, llm, tools=None):
        self.name = name
        self.role = role
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.crew_agent = CrewAgent(
            name=name,
            role=role,
            goal=description,
            backstory=description,
            llm=llm,
            tools=self.tools,
            verbose=True,
            max_iter=5,
            allow_delegation=True
        )

    def run(self, task):
        return self.crew_agent.run(task) 