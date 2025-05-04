import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Task, Crew

from agents.planner_agent import PlannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.reviewer_agent import ReviewerAgent
from agents.editor_agent import EditorAgent
from agents.seo_agent import SEOAgent
from agents.fact_checker_agent import FactCheckerAgent

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Set up the LLM and search tool
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
tools = [search_tool]

# Instantiate agents with LLM and tools
def build_agents():
    planner = PlannerAgent(llm=llm, tools=tools)
    researcher = ResearcherAgent(llm=llm, tools=tools)
    writer = WriterAgent(llm=llm, tools=tools)
    reviewer = ReviewerAgent(llm=llm, tools=tools)
    editor = EditorAgent(llm=llm, tools=tools)
    seo = SEOAgent(llm=llm, tools=tools)
    fact_checker = FactCheckerAgent(llm=llm, tools=tools)
    return [planner, researcher, writer, reviewer, editor, seo, fact_checker]

# Role-specific prompt templates
def get_prompt(agent_name, prev_output, user_query):
    prompts = {
        'Planner': f"Plan the structure and main points for a blog post about: '{user_query}'. Output a detailed outline.",
        'Researcher': f"Research the latest advancements, statistics, and trends for: '{user_query}'. Use the following outline as a guide: {prev_output}",
        'Writer': f"Write a comprehensive blog post based on this research and outline: {prev_output}",
        'Reviewer': f"Review the following blog post for factual accuracy, clarity, and coherence. Suggest improvements: {prev_output}",
        'Editor': f"Edit the following blog post for grammar, style, and readability. Make it engaging and professional: {prev_output}",
        'SEO Specialist': f"Optimize the following blog post for SEO. Suggest keywords, meta description, and improvements: {prev_output}",
        'Fact Checker': f"Fact-check the following blog post. Highlight any inaccuracies or unsupported claims: {prev_output}"
    }
    return prompts.get(agent_name, prev_output)

# Define the workflow pipeline using CrewAI's Task and Crew
def run_pipeline(user_query):
    agents = build_agents()
    input_data = user_query
    for agent in agents:
        print(f"\n--- {agent.name} is processing... ---")
        # Get a role-specific prompt for this agent
        prompt = get_prompt(agent.name, input_data, user_query)
        # Create a Task for this agent
        task = Task(
            description=prompt,
            agent=agent.crew_agent,
            expected_output=f"{agent.name} should produce a detailed and complete output for this step."
        )
        # Create a Crew with just this agent and task
        crew = Crew(
            agents=[agent.crew_agent],
            tasks=[task]
        )
        # Run the Crew and get the result
        input_data = crew.kickoff()
        print(f"\n{agent.name} output:\n{input_data}\n")
    return input_data

if __name__ == "__main__":
    user_query = "Write a comprehensive blog post about the impact of AI in Healthcare in 2024."
    print(f"\nRunning multi-agent pipeline for user query: {user_query}\n")
    final_output = run_pipeline(user_query)
    print("\n=== FINAL OUTPUT ===\n")
    print(final_output)
    with open("blog.md", "w", encoding="utf-8") as f:
        f.write(str(final_output))
    print("\nBlog article saved to blog.md.\n") 