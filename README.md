# AI-Powered Multi-Agent Content Studio (ReAct, CrewAI, Groq)

A Python project demonstrating a multi-agent, ReAct-style workflow for content generation using CrewAI, LangChain, and Groq LLMs. Agents are capable of reasoning, planning, executing, evaluating, and communicating with each other autonomously (no backend, no database, no web UI for now).

## Features
- Multi-agent orchestration (CrewAI, LangChain)
- ReAct (Reasoning + Action) agent pattern
- Agents can plan, execute, evaluate, and self-correct
- Uses Groq for fast, open-source LLM inference
- Easily extensible for new agent roles and tools

## Tech Stack
- **Python**
- **CrewAI** (multi-agent framework)
- **LangChain** (agent logic, tools, LLM integration)
- **Groq** (LLM provider)

## Setup Instructions

1. **Clone the repository**
2. **Create and activate a virtual environment**
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your Groq API key:**
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```
5. **Run the main script:**
   ```bash
   python main.py
   ```

## How it Works
- Agents (e.g., Researcher, Writer, Reviewer) are defined with roles, goals, and tools.
- Agents use the ReAct pattern to reason, act, observe, and iterate.
- Tasks are delegated, executed, and evaluated by agents without human intervention.
- Results are printed to the console.

---

This project is a prototype. Backend, database, and web UI will be added in future versions. 