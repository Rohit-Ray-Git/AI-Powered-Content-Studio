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
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Set up the LLM (Gemini 2.0 Flash) using CrewAI's LLM class
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
    reviewer.crew_agent.max_iter = 10
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
def get_prompt(agent_name, prev_output, user_query, content_type="Blog Post", script_length=None, language="English", subtopic=None): # Added language
    if agent_name == 'Researcher' and subtopic:
        return f"Research the following subtopic thoroughly for the main topic: '{user_query}'. Subtopic: {subtopic}. Use the web search tool to gather key facts, statistics, examples, and recent developments related to this subtopic. Provide a detailed summary paragraph (at least 10-15 sentences) covering the most important findings. Set the input parameter as : search_query."
    base_prompts = {
        'Planner': f"Plan the structure and main points for a blog post about: '{user_query}'. Use the web search tool if needed. When done, provide a final answer as a detailed outline.",
        'Researcher': f"Research the latest advancements, statistics, and trends for: '{user_query}'. Use the web search tool. Use the following outline as a guide: {prev_output}. When done, provide a final answer as a research summary.",
        'Writer': (
            f"Write a comprehensive, engaging, and descriptive blog post based on this research and outline: {prev_output}. "
            "For each section corresponding to a research subtopic, expand significantly on the provided research findings. Ensure each section has detailed context, multiple relevant examples, insightful analysis, and smooth transitions. Do not just list the research; elaborate on it thoroughly. "
            "**Focus on making the content highly engaging:** Use storytelling techniques, vivid descriptions, relatable analogies, and rhetorical questions to draw the reader in and maintain their interest throughout. Ensure a strong narrative flow. "
            "Where appropriate, identify key terms, statistics, or concepts and use the search tool to find relevant, authoritative URLs. Embed these as Markdown links (e.g., `link text`) directly within the text to support your points and provide further reading. Aim for 3-5 relevant links throughout the article. "
            "Aim for a substantial article, ideally over 2000 words. Ensure the content is rich and provides real value to the reader. "
            "Write in a conversational, informative tone suitable for Medium or professional blogs. "
            "Begin with a compelling introduction and end with a strong conclusion. "
            "Avoid bullet points except for short lists. Use paragraphs and storytelling. "
            "Provide the final output as a complete blog post in Markdown format. "
            "Here is an example of the desired style:\n\n"
            "Example:\n"
            "India's renewable energy journey is nothing short of remarkable. In 2024, the country achieved record-breaking growth, with solar panels gleaming atop rooftops from Mumbai to Chennai. This surge isn't just about numbers—it's about a nation embracing a cleaner, brighter future.\n\n"
        ),
        'Reviewer': f"Review the following blog post for factual accuracy, clarity, and coherence. Use the web search tool only if you need to verify a specific claim. After you have verified one or two claims, immediately write a review summary and end your response. Your output should be a single, concise review summary starting with 'Final Review:'. Do not output any more actions or thoughts after your summary. Here is the blog post: {prev_output}",
        'Editor': (
            "Edit the following blog post for grammar, style, and readability. **Crucially, enhance engagement:** Inject more vivid language, strengthen storytelling elements, improve flow, and ensure the tone is consistently captivating and professional. "
            "Review each section carefully. Elaborate on each section, add transitions, and ensure the post reads like a story, not just a list of facts. Pay close attention to the depth of each individual section based on the original research topics. "
            "Review existing links for relevance and quality. If appropriate, add 1-2 more high-quality, relevant Markdown links (`link text`) using the search tool to find authoritative sources for key claims or concepts that lack citation. Ensure links are integrated naturally. "
            "Significantly expand sections that seem brief or underdeveloped by adding more examples, context, narrative depth, or further explanation. Ensure the final post is comprehensive and feels complete (aiming for 2000+ words if the input is shorter). "
            "Ensure the final piece holds the reader's attention from start to finish. "
            f"Use paragraphs and storytelling effectively. Provide a final answer as the polished and expanded blog post in Markdown format: {prev_output}"
        ),
        'SEO Specialist': f"Take the following blog post (in Markdown format) and optimize it for SEO. Research relevant keywords using the search tool if necessary. Integrate keywords naturally, improve readability for search engines, and craft an optimized meta description (include it at the very beginning, like: META_DESCRIPTION: [Your description here]). Preserve existing Markdown links. Do NOT explain your process or plan. Your final output must be ONLY the complete, SEO-optimized blog post text in Markdown format, starting with the meta description line. Here is the blog post: {prev_output}",
        'Fact Checker': f"Fact-check the following blog post. Use the web search tool if needed. Highlight any inaccuracies or unsupported claims. Provide a final answer as a fact-check report: {prev_output}"
    }

    # --- Language Instruction ---
    language_instruction = f"\n\n**IMPORTANT: Generate the entire output text in {language}.**" if language != "English" else ""

    # --- Modify prompts based on content_type ---
    if agent_name == 'Writer':
        prompt = ""
        if content_type == "Social Media Posts":
            prompt = (
                f"Based on the following research summary about '{user_query}':\n{prev_output}\n\n"
                "Generate 3-5 distinct social media posts suitable for platforms like Twitter and LinkedIn.\n"
                "Each post should be:\n"
                "- **Informative & Descriptive:** Clearly explain a key insight, statistic, or finding from the research. Go beyond just stating the fact; add brief context or implication.\n"
                "- **Engaging & Attractive:** Use strong opening hooks, interesting questions, or compelling calls to action. Incorporate relevant emojis strategically to add visual appeal. Maintain a professional yet approachable and exciting tone.\n"
                "- **Concise:** Keep posts suitable for platform character limits (especially Twitter).\n"
                "- **Include relevant hashtags:** Use a mix of broad and specific hashtags.\n"
                "Format the output clearly, perhaps numbering each post or using separators like '---'.\n"
                "Provide the final output as the collection of social media posts in Markdown format."
            )
        elif content_type == "Video/Podcast Script":
            prompt = (
                f"Based on the following research summary about '{user_query}':\n{prev_output}\n\n"
                f"Write a detailed script suitable for an informative video or podcast segment, aiming for an approximate length of {script_length or 5} minutes (assume ~150 words per minute).\n"
                "Structure the script clearly:\n"
                "- Use standard script formatting (e.g., SCENE HEADING, NARRATOR/CHARACTER name in caps, Dialogue below name).\n"
                "- Break down content into logical scenes or sections (e.g., INT. STUDIO - DAY, SECTION 1: THE BASICS).\n"
                "- **Write dialogue/narration that is highly engaging:** Use storytelling, vivid language, rhetorical questions, and a conversational tone suitable for spoken delivery. Build interest and maintain listener attention.\n"
                "- Include clear action lines or descriptions for visuals where appropriate (e.g., [VISUAL: Graph showing market growth], [SOUND CUE: Upbeat intro music]).\n"
                "- Add speaker cues for pacing and emphasis (e.g., [PAUSE], [EMPHASIZE THIS POINT]).\n"
                "- **Focus on pacing:** Vary sentence structure and use pauses effectively to keep the listener engaged.\n"
                "- Ensure the content flows logically, covers the key research points effectively, and fits the target length.\n"
                "- Start with an engaging introduction and end with a clear conclusion or call to action.\n"
                "Provide the final output as the complete script in Markdown format."
            )
        # Default to Blog Post prompt if content_type is "Blog Post" or unrecognized
        else:
            prompt = base_prompts['Writer']
        return prompt + language_instruction # Add language instruction to writer prompt

    # --- Modify Reviewer prompt based on content_type ---
    if agent_name == 'Reviewer':
        prompt = "" # Initialize prompt variable
        if content_type == "Social Media Posts":
            prompt = (
                f"Review the following social media posts for clarity, engagement, tone, and hashtag relevance. "
                f"Provide brief feedback or suggestions if needed. If they look good, simply state that. "
                f"Do NOT start your response with 'Final Review:'. Here are the posts:\n{prev_output}"
            )
        elif content_type == "Video/Podcast Script":
             prompt = ( # Assign to prompt
                f"Review the following script for clarity, conversational flow, logical structure, and accuracy based on the original topic '{user_query}'. "
                f"Check if speaker/visual cues are used appropriately and consistently. Check if the pacing feels right for the target length. Provide constructive feedback on script elements, focusing on the {language} language if specified. "
                f"Do NOT start your response with 'Final Review:'. Here is the script:\n{prev_output}"
             )
        # Default to Blog Post review prompt
        else:
            prompt = base_prompts['Reviewer']
        return prompt # Reviewer doesn't generate main content, so language instruction might not be needed here, but feedback should consider it.

    # Add similar logic here for 'Editor' if its task needs to change significantly for scripts

    # --- Modify Editor prompt based on content_type ---
    if agent_name == 'Editor':
        prompt = ""
        if content_type == "Video/Podcast Script":
            prompt = (
                f"Edit the following script for grammar, style, clarity, and conciseness, focusing on spoken language. **Significantly enhance engagement:** Inject more vivid descriptions, strengthen narrative elements, improve pacing, and ensure the tone is consistently captivating. "
                f"Ensure consistent formatting for scene headings, character names, dialogue, and cues. Smooth out awkward phrasing. Check for flow and pacing. "
                f"Provide the final output as the polished script in Markdown format: {prev_output}"
            )
        # Default to Blog Post editor prompt
        else: # Added else block
             prompt = base_prompts['Editor']
        return prompt + language_instruction # Add language instruction to editor prompt

    # --- Add language instruction for other relevant agents ---
    # SEO might need language context, Fact Checker might too. For now, focus on Writer/Editor.
    # if agent_name in ['SEO Specialist', 'Fact Checker']:
    #     return base_prompts.get(agent_name, f"Process the following input: {prev_output}") + language_instruction

    # Return base prompt for other agents or fallback
    return base_prompts.get(agent_name, f"Process the following input: {prev_output}")

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
        # Use all non-empty lines if outline parsing fails
        subtopics = [line for line in lines if line and len(line) > 3] or [user_query] # Ensure at least the original query if all else fails
    return subtopics

def plan_task(user_query, planner_agent, callback=None):
    # Use the planner agent to break down the user_query into subtasks
    prompt = get_prompt('Planner', '', user_query) # prev_output is empty for planner
    task = Task(
        description=prompt,
        agent=planner_agent.crew_agent,
        expected_output="A detailed outline or list of subtopics for the blog post."
    )
    crew = Crew(agents=[planner_agent.crew_agent], tasks=[task])
    result = crew.kickoff()
    if callback:
        callback(f"Planner Result:\n{result}")
    if isinstance(result, str):
        subtopics = extract_subtopics_from_outline(result)
        if len(subtopics) >= 3:
            if callback:
                callback(f"Extracted Subtopics: {', '.join(subtopics)}")
            return subtopics
    # If planner output is not usable, just use the whole query as one subtopic
    if callback:
        callback(f"Could not extract distinct subtopics. Using full query: '{user_query}'")
    return [user_query]

def run_research_subtasks(user_query, subtopics, researcher_agent, callback=None):
    results = {}
    for subtopic in subtopics:
        if callback:
            callback(f"\n--- Researching: {subtopic} ---")
        prompt = get_prompt('Researcher', '', user_query, subtopic=subtopic)
        task = Task(
            description=prompt,
            agent=researcher_agent.crew_agent,
            expected_output=f"A detailed summary paragraph (10 to 15 sentences minimum) of research findings for the subtopic: {subtopic}." # Corrected expected output length
        )
        crew = Crew(agents=[researcher_agent.crew_agent], tasks=[task])
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = crew.kickoff()
                results[subtopic] = result
                if callback:
                    callback(f"\nResearch Result for '{subtopic}':\n{result}\n")
                break
            except Exception as e:
                error_msg = str(e)
                if 'RESOURCE_EXHAUSTED' in error_msg or 'RateLimitError' in error_msg:
                    # Try to extract suggested retry delay from error message
                    retry_delay = 20
                    match = re.search(r'"retryDelay":\s*"(\\d+)s"', error_msg)
                    if match:
                        retry_delay = int(match.group(1))
                    message = f"Rate limit hit. Waiting {retry_delay} seconds before retrying (attempt {attempt+1}/{max_retries})..."
                    if callback: callback(message)
                    else: print(message)
                    time.sleep(retry_delay)
                else:
                    message = f"Error during research for subtopic '{subtopic}': {e}"
                    if callback: callback(message)
                    else: print(message)
                    results[subtopic] = f"Error: {e}"
                    break
        else:
            results[subtopic] = "Error: Rate limit exceeded after retries."
    return results

def get_trending_topics(search_tool, num_topics=6, callback=None):
    """Fetches current trending topics using the search tool."""
    if callback:
        callback("Fetching trending topics...")
    try:
        # Use a query likely to return a list or summary of trends
        # Updated query to focus on new tech and inventions
        search_query = f"list the top {num_topics} recent breakthroughs or new inventions in technology"
        # Pass the query using the expected keyword argument 'search_query'
        trends_result = search_tool.run(search_query=search_query)
        if callback:
            callback(f"Raw trends result: {trends_result[:200]}...") # Log raw result

        # --- Handle potential dictionary output from search tool ---
        text_content = ""
        if isinstance(trends_result, dict):
            # Try common keys where relevant text might be found
            if 'answerBox' in trends_result and 'snippet' in trends_result['answerBox']:
                text_content = trends_result['answerBox'].get('title', '') + "\n" + trends_result['answerBox'].get('snippet', '')
            elif 'organic' in trends_result and isinstance(trends_result['organic'], list):
                # Combine snippets from organic results
                text_content = "\n".join([item.get('snippet', '') for item in trends_result['organic'] if 'snippet' in item])
            else:
                # Fallback: convert the whole dict to string if specific keys aren't found
                text_content = str(trends_result)
        elif isinstance(trends_result, str):
            text_content = trends_result
        else: # Handle other unexpected types
            text_content = str(trends_result)
        # --- End handling dictionary output ---

        # --- Improved Parsing Logic ---
        # 1. Remove common introductory/concluding phrases (case-insensitive)
        text_content = re.sub(r'(?:Here are|Here\'s|list of|Explore the archives|This is the \d+.. year).*?\n', '', text_content, flags=re.IGNORECASE | re.DOTALL)
        text_content = re.sub(r'didn\'t make the cut.*', '', text_content, flags=re.IGNORECASE | re.DOTALL)

        # 2. Split by common list separators (newline, semicolon, bullet, number followed by dot/parenthesis)
        potential_topics = re.split(r'\s*(?:\n|;|\d+[.)]|\*|·|-)\s*', text_content)
        topics = []

        for item in potential_topics:
            # 3. Clean each item aggressively
            topic_text = item.strip()
            # Remove any remaining leading list markers just in case
            topic_text = re.sub(r'^\s*(?:\d+[.)]?|\*|·|-)\s*', '', topic_text).strip()
            # Remove trailing punctuation/junk
            topic_text = topic_text.rstrip('.,;:').strip()

            # Filter out empty strings and very short items
            if topic_text and 1 < len(topic_text.split()) < 10: # Keep word count filter reasonable
                 if topic_text not in topics: # Avoid adding duplicates
                    topics.append(topic_text)

        return topics[:num_topics] # Return the requested number of topics
    except Exception as e:
        if callback: callback(f"Error fetching trending topics: {e}")
        else: print(f"Error fetching trending topics: {e}")
        return [] # Return empty list on error

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
def run_pipeline(user_query, content_type="Blog Post", script_length=None, language="English", callback=None): # Added language
    if callback:
        callback(f"Starting pipeline for query: {user_query}")
    agents = build_agents()
    if callback: callback("Running Planner...")
    subtopics = plan_task(user_query, agents['planner'], callback=callback)
    research_results = run_research_subtasks(user_query, subtopics, agents['researcher'])
    research_summary = aggregate_research_results(research_results)
    input_data = research_summary
    final_content = None # Renamed from blog_post to be more generic
    fact_check_report = None

    # --- Adjust agent sequence based on content_type ---
    if content_type == "Blog Post":
        agent_sequence = ['writer', 'reviewer', 'editor', 'seo', 'fact_checker']
    elif content_type == "Social Media Posts":
        # Shorter sequence for social media
        agent_sequence = ['writer'] # Only run the writer for social media posts
    elif content_type == "Video/Podcast Script":
        # Sequence for scripts
        agent_sequence = ['writer', 'reviewer', 'editor'] # Maybe skip seo/fact-check for scripts
    else: # Default to Blog Post sequence
        agent_sequence = ['writer', 'reviewer', 'editor', 'seo', 'fact_checker']

    for agent_key in agent_sequence:
        agent = agents[agent_key]
        prompt = get_prompt(agent.name, input_data, user_query, content_type=content_type, script_length=script_length, language=language) # Pass language
        if callback: # Check if callback exists before calling
            callback(f"[DEBUG] Prompt for {agent.name} (start): {prompt[:200]}...") # Log start of prompt
        task = Task(
            description=prompt,
            agent=agent.crew_agent,
            expected_output=f"A detailed and complete output representing the result of the {agent.name}'s task (e.g., written blog post, review summary, edited post, SEO suggestions, fact-check report)."
        )
        crew = Crew(agents=[agent.crew_agent], tasks=[task])
        result = crew.kickoff()
        if callback:
            callback(f"[DEBUG] Output from {agent.name} (start):\n{repr(str(result)[:300])}...\n") # Log start of result
        # Check for placeholder output
        if isinstance(result, str) and ("I am ready to edit the blog post once it is provided." in result or len(result.strip()) == 0):
            message = f"\n[ERROR] {agent.name} did not receive valid input or gave placeholder output. Pipeline stopped.\n"
            if callback: callback(message)
            return None, None # Return None for both outputs

        # Store the final output from the *last* agent in the sequence
        if agent_key == agent_sequence[-1] or (agent_key == 'seo' and content_type == "Blog Post"): # Special case for SEO in blog posts
            final_content = result

        if agent_key == 'fact_checker':
            fact_check_report = result

        input_data = result # Pass output to the next agent in the sequence
        if isinstance(result, str) and "Agent stopped due to iteration limit or time limit" in result:
            message = f"\n[ERROR] {agent.name} failed to complete its task (limit reached). Pipeline stopped."
            if callback: callback(message)
            return None, None # Return None for both outputs
    if callback:
        callback("\n--- Pipeline finished ---")
    return final_content, fact_check_report # Return the final generated content and fact check report

def extract_body_content(html):
    match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return html.strip()

if __name__ == "__main__":
    user_query = "Write a detailed report on the impact of AI in human life."
    print(f"\nRunning multi-agent pipeline via main script for query: {user_query}\n")
    # Pass print function as a simple callback when running directly
    final_content, fact_check_report = run_pipeline(user_query, content_type="Blog Post", language="English", callback=print) # Specify language
    print("\n=== FINAL OUTPUT ===\n")
    print("Type of final_content:", type(final_content))
    print("repr(final_content) (first 200 chars):", repr(str(final_content)[:200]))
    # Clean code block markers before saving
    cleaned_content_text = clean_code_blocks(str(final_content))

    # Save as Markdown
    with open("blog.md", "w", encoding="utf-8") as f:
        f.write(cleaned_content_text)
    print("\nBlog article saved to blog.md as Markdown.\n")

    # Convert Markdown to HTML
    html_content = markdown.markdown(cleaned_content_text, extensions=['fenced_code', 'tables'])

    # Create a basic HTML structure
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AI Generated Blog Post</title></head>
<body>{html_content}</body>
</html>"""

    with open("blog.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    print("\nBlog article converted to HTML and saved to blog.html.\n")
    if fact_check_report and isinstance(fact_check_report, str):
        with open("fact_check_report.md", "w", encoding="utf-8") as f:
            f.write(fact_check_report)
        print("\nFact-check report saved to fact_check_report.md.\n")
