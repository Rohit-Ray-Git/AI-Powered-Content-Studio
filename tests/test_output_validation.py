import unittest
from unittest.mock import patch
from main import run_pipeline

class TestOutputValidation(unittest.TestCase):
    @patch('main.WriterAgent')
    @patch('main.build_agents')
    def test_writer_agent_bad_output(self, mock_build_agents, mock_writer_agent):
        # Mock the writer agent to return a placeholder output
        class DummyAgent:
            def __init__(self, *args, **kwargs):
                self.name = 'Writer'
                self.crew_agent = self
            def run(self, task):
                return "I am ready to edit the blog post once it is provided."
        # Patch build_agents to use the dummy writer agent
        agents = {
            'planner': DummyAgent(),
            'researcher': DummyAgent(),
            'writer': DummyAgent(),
            'reviewer': DummyAgent(),
            'editor': DummyAgent(),
            'seo': DummyAgent(),
            'fact_checker': DummyAgent(),
        }
        mock_build_agents.return_value = agents
        # Run the pipeline with valid input
        result = run_pipeline('Test topic', content_type='Blog Post', language='English', tone='Informational')
        self.assertEqual(result, (None, None))

if __name__ == '__main__':
    unittest.main() 