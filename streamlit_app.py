"""
Streamlit Frontend for the AI-Powered Content Studio.

Allows users to input a topic, run the content generation pipeline,
and view/download the results.
"""

import streamlit as st
import os
import sys
import traceback
import streamlit.components.v1 as components
import markdown # Import the markdown library

# Ensure the main script's directory is in the path to find modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your pipeline and helper functions
try:
    from main import run_pipeline, clean_code_blocks, get_trending_topics, search_tool
except ImportError as e:
    st.error(f"Error importing functions from main.py: {e}")
    st.error(f"Project Root: {project_root}")
    st.error(f"Sys Path: {sys.path}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Content Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- App Title and Description ---
st.title("✍️ AI-Powered Content Studio")
st.markdown("Generate comprehensive blog posts on any topic using a multi-agent AI pipeline.")
st.markdown("---")

# --- Custom CSS for better formatting ---
st.markdown("""
<style>
.stMarkdown {
    line-height: 1.6;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}
.stMarkdown p {
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# --- Input Section ---
st.header("Choose or Enter Your Topic")

# --- Trending Topics Section ---
@st.cache_data(ttl=3600)
def fetch_trends():
    try:
        topics = get_trending_topics(search_tool, num_topics=6)
        return topics
    except Exception as e:
        print(f"Error in fetch_trends: {e}")
        return []

trending_topics = fetch_trends()

# --- DEBUG line removed ---
# st.write("DEBUG: Fetched trending_topics list:", trending_topics)

if trending_topics:
    st.subheader("🔥 Trending Topics (Click to select)")

    if 'user_query_input' not in st.session_state:
        st.session_state.user_query_input = ""

    num_columns = 3
    rows = [trending_topics[i:i + num_columns] for i in range(0, len(trending_topics), num_columns)]

    for row in rows:
        cols = st.columns(num_columns)
        for idx, topic in enumerate(row):
            with cols[idx]:
                with st.container(border=True):
                    if st.button(topic, key=f"topic_{topic}", use_container_width=True):
                        st.session_state.user_query_input = topic
                        st.rerun()
                    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# --- Topic Input ---
user_query = st.text_area(
    "Describe the topic you want the AI to write about:",
    height=100,
    placeholder="e.g., The future of renewable energy in India, or Explain the basics of quantum computing",
    key='user_query_input'
)

# Initialize session state for blog content if it doesn't exist
if 'blog_post_content' not in st.session_state:
    st.session_state.blog_post_content = None
if 'fact_check_report' not in st.session_state:
    st.session_state.fact_check_report = None

# --- Run Button ---
st.markdown("<br>", unsafe_allow_html=True)
col_run_button, _ = st.columns([1, 3])
with col_run_button:
    run_button = st.button("🚀 Generate Content", type="primary", use_container_width=True)


if run_button:
    if not st.session_state.get('user_query_input', ''):
        st.warning("Please enter a topic description above.")
    else:
        with st.spinner("🤖 The AI agents are working... This may take several minutes..."):
            try:
                query_to_run = st.session_state.user_query_input
                blog_post_content, fact_check_report = run_pipeline(query_to_run, callback=None)

                # Store results in session state
                st.session_state.blog_post_content = blog_post_content
                st.session_state.fact_check_report = fact_check_report

                if blog_post_content:
                    st.success("✅ Content generation complete!")
                else:
                    st.error("😔 Content generation failed or returned empty. Please check the console logs for errors.")
                    st.session_state.blog_post_content = None # Clear on failure
                    st.session_state.fact_check_report = None
            except Exception as e:
                st.error("An error occurred during the pipeline execution:")
                st.error(traceback.format_exc())
                st.session_state.blog_post_content = None # Clear on error
                st.session_state.fact_check_report = None

# --- Display Area (Moved outside the 'if run_button' block) ---
st.markdown("---")
st.header("Generated Content")

# Display content if it exists in session state
if st.session_state.get('blog_post_content'):
    # Display the blog post preview (rendered from Markdown)
    st.subheader("📄 Blog Post Preview (Rendered from Markdown)")
    st.markdown(clean_code_blocks(str(st.session_state.blog_post_content)), unsafe_allow_html=True)

    # Provide download buttons
    st.subheader("💾 Download Files")
    col1, col2, col3 = st.columns(3)

    # --- Prepare Markdown Download Data ---
    try:
        cleaned_md_content = clean_code_blocks(str(st.session_state.blog_post_content))
        col1.download_button(
            label="Download Markdown (.md)",
            data=cleaned_md_content,
            file_name="blog_post.md",
            mime="text/markdown",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error preparing Markdown download data: {e}")

    # --- Prepare HTML Download Data ---
    try:
        # Convert Markdown from session state to HTML
        html_content = markdown.markdown(cleaned_md_content)
        # Basic HTML structure
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog Post</title>
    <style> body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }} </style>
</head>
<body>
{html_content}
</body>
</html>"""
        col2.download_button(
            label="Download HTML (.html)",
            data=full_html,
            file_name="blog_post.html",
            mime="text/html",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error preparing HTML download data: {e}")

    # --- Prepare Fact Check Report Download Data ---
    if st.session_state.get('fact_check_report'):
        try:
            st.markdown("---")
            st.subheader("🧐 Fact-Check Report")
            st.markdown(str(st.session_state.fact_check_report)) # Display report
            fact_check_data = str(st.session_state.fact_check_report)
            col3.download_button(
                label="Download Fact-Check Report (.md)",
                data=fact_check_data,
                file_name="fact_check_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error preparing Fact Check download data: {e}")

elif run_button and not st.session_state.get('user_query_input', ''):
    # This case is handled inside the main 'if run_button' block now
    pass # No need for separate handling here
elif not run_button and not st.session_state.get('blog_post_content'):
    # Show initial message only if not run and no content exists
    st.info("Enter a topic or select a trending one, then click 'Generate Content' to start.")
