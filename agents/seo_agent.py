"""
SEO agent for Agent Studio prototype.
"""

from .base_agent import BaseAgent

class SEOAgent(BaseAgent):
    def __init__(self, llm, tools=None, name="SEO Specialist", description="Responsible for optimizing content for search engines."):
        super().__init__(name, role="SEO Specialist", description=description, llm=llm, tools=tools) 