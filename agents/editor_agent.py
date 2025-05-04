"""
Editor agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class EditorAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="Editor", description="Responsible for editing and polishing content for clarity, grammar, and style."):
        super().__init__(name, role="Editor", description=description, llm=llm, tools=tools) 