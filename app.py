"""
Streamlit UI for Agent Studio prototype.
"""

import streamlit as st
from agents import ResearcherAgent, WriterAgent, ReviewerAgent, EditorAgent, SEOAgent, FactCheckerAgent, PlannerAgent
from workflow import Workflow

st.title("🧑‍💻 Agent Studio Prototype")

# Sidebar: Agent creation
st.sidebar.header("Create Agents")
if 'agents' not in st.session_state:
    st.session_state['agents'] = []

agent_type = st.sidebar.selectbox("Agent Type", [
    "Researcher", "Writer", "Reviewer", "Editor", "SEO Specialist", "Fact Checker", "Planner"
])
agent_name = st.sidebar.text_input("Agent Name")
agent_desc = st.sidebar.text_area("Agent Description")
if st.sidebar.button("Add Agent"):
    if agent_name and agent_desc:
        if agent_type == "Researcher":
            agent = ResearcherAgent(name=agent_name, description=agent_desc)
        elif agent_type == "Writer":
            agent = WriterAgent(name=agent_name, description=agent_desc)
        elif agent_type == "Reviewer":
            agent = ReviewerAgent(name=agent_name, description=agent_desc)
        elif agent_type == "Editor":
            agent = EditorAgent(name=agent_name, description=agent_desc)
        elif agent_type == "SEO Specialist":
            agent = SEOAgent(name=agent_name, description=agent_desc)
        elif agent_type == "Fact Checker":
            agent = FactCheckerAgent(name=agent_name, description=agent_desc)
        elif agent_type == "Planner":
            agent = PlannerAgent(name=agent_name, description=agent_desc)
        st.session_state['agents'].append(agent)
        st.sidebar.success(f"{agent_type} '{agent_name}' added!")
    else:
        st.sidebar.error("Please fill all fields.")

# Main: Workflow creation
st.header("Create Workflow")
workflow_name = st.text_input("Workflow Name", value="My Workflow")
if 'workflow' not in st.session_state or st.session_state['workflow'].name != workflow_name:
    st.session_state['workflow'] = Workflow(workflow_name)

# Add agents to workflow
st.subheader("Add Agents to Workflow")
agent_names = [a.name for a in st.session_state['agents']]
selected_agents = st.multiselect("Select Agents", agent_names)
if st.button("Add Selected Agents"):
    st.session_state['workflow'].agents = [a for a in st.session_state['agents'] if a.name in selected_agents]
    st.success("Agents added to workflow.")

# Add tasks
st.subheader("Add Tasks")
if 'tasks' not in st.session_state:
    st.session_state['tasks'] = []
task_input = st.text_area("Task Description")
if st.button("Add Task"):
    if task_input:
        st.session_state['tasks'].append(task_input)
        st.success("Task added.")
    else:
        st.error("Please enter a task description.")

# Assign tasks to workflow
st.session_state['workflow'].tasks = st.session_state['tasks']

# Run workflow
if st.button("Run Workflow"):
    if not st.session_state['workflow'].agents:
        st.error("Add at least one agent to the workflow.")
    elif not st.session_state['workflow'].tasks:
        st.error("Add at least one task to the workflow.")
    else:
        results = st.session_state['workflow'].run()
        st.header("Results")
        for agent_name, task, result in results:
            st.markdown(f"**Agent:** {agent_name}")
            st.markdown(f"**Task:** {task}")
            st.markdown(f"**Result:** {result}")
            st.markdown("---") 