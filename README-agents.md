# Agent Studio Prototype

This directory contains a minimal, stepwise prototype for an AI-powered content studio using autonomous agents.

## Features
- No complex backend (no Django, FastAPI, or database)
- Streamlit UI for interaction
- Agents powered by LangChain, Groq API, CrewAI, etc.
- In-memory workflow and agent management
- Modular and extensible for future backend/database integration

## Structure
- `app.py` — Streamlit UI
- `agents.py` — Agent creation and logic
- `workflow.py` — Workflow management
- `utils.py` — Utility functions (API calls, etc.)
- `requirements.txt` — Python dependencies

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py` 