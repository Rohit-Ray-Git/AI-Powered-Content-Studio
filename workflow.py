"""
Workflow management for Agent Studio prototype.
"""

class Workflow:
    def __init__(self, name):
        self.name = name
        self.agents = []
        self.tasks = []
        self.results = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        # Simple round-robin assignment for now
        for i, task in enumerate(self.tasks):
            agent = self.agents[i % len(self.agents)]
            result = agent.run(task)
            self.results.append((agent.name, task, result))
        return self.results 