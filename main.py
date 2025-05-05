import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import SerperDevTool
from crewai import Task, Crew, LLM
import time
import re
import markdown

from agents.planner_agent import PlannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.reviewer_agent import ReviewerAgent
from agents.editor_agent import EditorAgent
from agents.seo_agent import SEOAgent
from agents.fact_checker_agent import FactCheckerAgent

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or GOOGLE_API_KEY
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Set up the LLM (Gemini 2.5 Pro Preview) using CrewAI's LLM class
llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2
)

# Set up the search tool
search_tool = SerperDevTool()
tools = [search_tool]

# Helper to add the search_query instruction to backstory
def add_search_query_instruction(text):
    return f"{text}\n\nSet the input parameter as : search_query."

# Subtopics for static research tasks
STATIC_RESEARCH_SUBTOPICS = [
    "Market size and growth",
    "AI in diagnostics",
    "AI in drug discovery",
    "AI in personalized medicine",
    "AI in healthcare administration",
    "AI in public health",
    "AI in robotics/surgery",
    "AI in mental health",
    "Key challenges and ethical issues",
    "Future trends"
]

# Instantiate agents with LLM and tools, set max_iter and max_execution_time
def build_agents():
    planner = PlannerAgent(llm=llm, tools=tools, description=add_search_query_instruction("Responsible for planning the workflow and assigning tasks."))
    planner.crew_agent.max_iter = 10
    planner.crew_agent.max_execution_time = 60
    researcher = ResearcherAgent(
        llm=llm,
        tools=tools,
        description=add_search_query_instruction(
            "Responsible for researching topics and gathering information. For each subtopic, do a maximum of 1 web search and summarize your findings in 2-3 concise bullet points."
        )
    )
    researcher.crew_agent.max_iter = 10
    researcher.crew_agent.max_execution_time = 60
    writer = WriterAgent(llm=llm, tools=tools, description=add_search_query_instruction("Responsible for writing content based on research."))
    writer.crew_agent.max_iter = 10
    writer.crew_agent.max_execution_time = 60
    reviewer = ReviewerAgent(
        llm=llm,
        tools=tools,
        description=add_search_query_instruction(
            "Responsible for reviewing and providing feedback on content. Review the blog post for factual accuracy, clarity, and coherence. Use the web search tool only if you need to verify a specific claim. After you have verified one or two claims, immediately write a review summary and end your response. Your output should be a single, concise review summary starting with 'Final Review:'. Do not output any more actions or thoughts after your summary."
        )
    )
    reviewer.crew_agent.max_iter = 2
    reviewer.crew_agent.max_execution_time = 60
    editor = EditorAgent(llm=llm, tools=tools, description=add_search_query_instruction("Responsible for editing and polishing content for clarity, grammar, and style."))
    editor.crew_agent.max_iter = 10
    editor.crew_agent.max_execution_time = 60
    seo = SEOAgent(llm=llm, tools=tools, description=add_search_query_instruction("Responsible for optimizing content for search engines."))
    seo.crew_agent.max_iter = 10
    seo.crew_agent.max_execution_time = 60
    fact_checker = FactCheckerAgent(llm=llm, tools=tools, description=add_search_query_instruction("Responsible for verifying the accuracy of information."))
    fact_checker.crew_agent.max_iter = 10
    fact_checker.crew_agent.max_execution_time = 60
    return {
        'planner': planner,
        'researcher': researcher,
        'writer': writer,
        'reviewer': reviewer,
        'editor': editor,
        'seo': seo,
        'fact_checker': fact_checker
    }

# Role-specific prompt templates (explicitly instruct to use the search tool and provide a final answer)
def get_prompt(agent_name, prev_output, user_query, subtopic=None):
    if agent_name == 'Researcher' and subtopic:
        return f"Research the following subtopic for: '{user_query}'. Subtopic: {subtopic}. Use the web search tool if needed. Summarize your findings in 2-3 concise bullet points. Set the input parameter as : search_query."
    prompts = {
        'Planner': f"Plan the structure and main points for a blog post about: '{user_query}'. Use the web search tool if needed. When done, provide a final answer as a detailed outline. Set the input parameter as : search_query.",
        'Researcher': f"Research the latest advancements, statistics, and trends for: '{user_query}'. Use the web search tool. Use the following outline as a guide: {prev_output}. When done, provide a final answer as a research summary. Set the input parameter as : search_query.",
        'Writer': (
            "Write a comprehensive, engaging, and descriptive blog post based on this research and outline: {prev_output}. "
            "For each section, expand on the points with context, examples, and smooth transitions. "
            "Write in a conversational, informative tone suitable for Medium or professional blogs. "
            "Begin with a compelling introduction and end with a strong conclusion. "
            "Avoid bullet points except for short lists. Use paragraphs and storytelling. "
            "Here is an example of the desired style:\n\n"
            "Example:\n"
            "India's renewable energy journey is nothing short of remarkable. In 2024, the country achieved record-breaking growth, with solar panels gleaming atop rooftops from Mumbai to Chennai. This surge isn't just about numbers—it's about a nation embracing a cleaner, brighter future.\n\n"
            "Set the input parameter as : search_query."
        ),
        'Reviewer': f"Review the following blog post for factual accuracy, clarity, and coherence. Use the web search tool only if you need to verify a specific claim. After you have verified one or two claims, immediately write a review summary and end your response. Your output should be a single, concise review summary starting with 'Final Review:'. Do not output any more actions or thoughts after your summary. Here is the blog post: {prev_output} Set the input parameter as : search_query.",
        'Editor': (
            "Edit the following blog post for grammar, style, and readability. Make it more engaging and professional. "
            "Elaborate on each section, add transitions, and ensure the post reads like a story, not a list. "
            "If any section is too brief, expand it with examples, context, or narrative. "
            "Use paragraphs and storytelling. Provide a final answer as the edited blog post: {prev_output} Set the input parameter as : search_query."
        ),
        'SEO Specialist': f"Optimize the following blog post for SEO. Use the web search tool if needed. Suggest keywords, meta description, and improvements. Provide a final answer as an SEO-optimized version: {prev_output} Set the input parameter as : search_query.",
        'Fact Checker': f"Fact-check the following blog post. Use the web search tool if needed. Highlight any inaccuracies or unsupported claims. Provide a final answer as a fact-check report: {prev_output} Set the input parameter as : search_query."
    }
    return prompts.get(agent_name, prev_output)

def extract_subtopics_from_outline(outline):
    # Try to extract subtopics from the planner's outline (numbered, bulleted, or indented lines)
    lines = [line.strip() for line in outline.split('\n') if line.strip()]
    subtopics = []
    for line in lines:
        # Match lines that look like section headers or main points
        if re.match(r'^(\*+|\d+\.|[A-ZIVX]+\.|-)', line):
            # Remove leading bullets/numbers and whitespace
            clean = re.sub(r'^(\*+|\d+\.|[A-ZIVX]+\.|-)+\s*', '', line)
            # Only add if not too short
            if len(clean) > 3:
                subtopics.append(clean)
        # Or lines that look like section headers (e.g., 'II. Current Status...')
        elif re.match(r'^[A-Z][^a-z]+\.', line):
            subtopics.append(line.strip('. '))
    # Fallback: if not enough subtopics, use lines that look like section headers
    if len(subtopics) < 3:
        subtopics = [line for line in lines if line and len(line) > 3]
    return subtopics

def plan_task(user_query, planner_agent):
    # Use the planner agent to break down the user_query into subtasks
    prompt = get_prompt('Planner', '', user_query)
    task = Task(
        description=prompt,
        agent=planner_agent.crew_agent,
        expected_output="Planner should produce a list of subtasks or sections."
    )
    crew = Crew(agents=[planner_agent.crew_agent], tasks=[task])
    result = crew.kickoff()
    if isinstance(result, str):
        subtopics = extract_subtopics_from_outline(result)
        if len(subtopics) >= 3:
            return subtopics
    # If planner output is not usable, just use the whole query as one subtopic
    return [user_query]

def run_research_subtasks(user_query, subtopics, researcher_agent):
    results = {}
    for subtopic in subtopics:
        print(f"\n--- Researching: {subtopic} ---")
        prompt = get_prompt('Researcher', '', user_query, subtopic=subtopic)
        task = Task(
            description=prompt,
            agent=researcher_agent.crew_agent,
            expected_output=f"Researcher should produce a concise summary for subtopic: {subtopic}."
        )
        crew = Crew(agents=[researcher_agent.crew_agent], tasks=[task])
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = crew.kickoff()
                results[subtopic] = result
                print(f"\n{subtopic} output:\n{result}\n")
                break
            except Exception as e:
                error_msg = str(e)
                if 'RESOURCE_EXHAUSTED' in error_msg or 'RateLimitError' in error_msg:
                    # Try to extract suggested retry delay from error message
                    retry_delay = 20
                    match = re.search(r'"retryDelay":\s*"(\\d+)s"', error_msg)
                    if match:
                        retry_delay = int(match.group(1))
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retrying (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error during research for subtopic '{subtopic}': {e}")
                    results[subtopic] = f"Error: {e}"
                    break
        else:
            print(f"Failed to complete research for subtopic '{subtopic}' after {max_retries} attempts.")
            results[subtopic] = "Error: Rate limit exceeded after retries."
    return results

def aggregate_research_results(results):
    # Combine all subtopic results into a single research summary
    summary = "\n\n".join([f"**{subtopic}:**\n{output}" for subtopic, output in results.items()])
    return summary

def clean_code_blocks(text):
    # Remove all code block markers (``` and ```html) from the start and end
    cleaned = re.sub(r'^```(?:html)?\s*', '', text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

# Define the workflow pipeline using CrewAI's Task and Crew
def run_pipeline(user_query):
    print(f"[DEBUG] User query at pipeline start: {user_query}")
    agents = build_agents()
    subtopics = plan_task(user_query, agents['planner'])
    research_results = run_research_subtasks(user_query, subtopics, agents['researcher'])
    research_summary = aggregate_research_results(research_results)
    input_data = research_summary
    blog_post = None
    fact_check_report = None

    for agent_key in ['writer', 'reviewer', 'editor', 'seo', 'fact_checker']:
        agent = agents[agent_key]
        print(f"\n--- {agent.name} is processing... ---")
        prompt = get_prompt(agent.name, input_data, user_query)
        print(f"[DEBUG] Prompt for {agent.name}: {prompt[:200]}")
        task = Task(
            description=prompt,
            agent=agent.crew_agent,
            expected_output=f"{agent.name} should produce a detailed and complete output for this step. Set the input parameter as : search_query."
        )
        crew = Crew(agents=[agent.crew_agent], tasks=[task])
        result = crew.kickoff()
        print(f"\n[DEBUG] {agent.name} output (first 500 chars):\n{repr(str(result)[:500])}\n")
        # Check for placeholder output
        if isinstance(result, str) and ("I am ready to edit the blog post once it is provided." in result or len(result.strip()) == 0):
            print(f"\n[ERROR] {agent.name} did not receive valid input. Pipeline stopped.\n")
            return None, None
        if agent_key == 'seo':
            blog_post = result  # Save the SEO-optimized blog post
        if agent_key == 'fact_checker':
            fact_check_report = result
        # Only update input_data for the next agent if not the fact_checker
        if agent_key != 'fact_checker':
            input_data = result
        if isinstance(result, str) and "Agent stopped due to iteration limit or time limit" in result:
            print(f"\nERROR: {agent.name} failed to complete its task. Pipeline stopped.")
            return None, None
    return blog_post, fact_check_report

def extract_body_content(html):
    match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return html.strip()

if __name__ == "__main__":
    user_query = "Write a detailed report on the impact of renewable energy adoption in India in 2024."
    print(f"\nRunning multi-agent pipeline for user query: {user_query}\n")
    blog_post, fact_check_report = run_pipeline(user_query)
    print("\n=== FINAL OUTPUT ===\n")
    print(blog_post)
    print("Type of blog_post:", type(blog_post))
    print("repr(blog_post) (first 200 chars):", repr(str(blog_post)[:200]))
    # Clean code block markers before saving
    cleaned_blog_post = clean_code_blocks(str(blog_post))
    with open("blog.html", "w", encoding="utf-8") as f:
        f.write(cleaned_blog_post)
    print("\nBlog article saved to blog.html as HTML (code blocks removed).\n")
    if fact_check_report and isinstance(fact_check_report, str):
        with open("fact_check_report.md", "w", encoding="utf-8") as f:
            f.write(fact_check_report)
        print("\nFact-check report saved to fact_check_report.md.\n") 
        