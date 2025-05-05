"""
Streamlit Frontend for the AI-Powered Content Studio.

Allows users to input a topic, run the content generation pipeline,
and view/download the results.
"""

import streamlit as st
import os
import sys
import traceback
import streamlit.components.v1 as components # Import components

# Ensure the main script's directory is in the path to find modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main pipeline function AFTER setting the path
try:
    # Assuming main.py is in the same directory as this streamlit_app.py
    from main import run_pipeline, clean_code_blocks
except ImportError as e:
    st.error(f"Error importing functions from main.py: {e}")
    st.error(f"Project Root: {project_root}")
    st.error(f"Sys Path: {sys.path}")
    st.stop() # Stop execution if import fails

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
    line-height: 1.6; /* Improve text readability */
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    margin-top: 1.5em; /* Add space above headers */
    margin-bottom: 0.5em;
}
.stMarkdown p {
    margin-bottom: 1em; /* Add space between paragraphs */
}
</style>
""", unsafe_allow_html=True)

# --- Input Section ---
st.header("1. Enter Your Topic")
user_query = st.text_area(
    "Describe the topic you want the AI to write about:",
    height=100,
    placeholder="e.g., The future of renewable energy in India, or Explain the basics of quantum computing"
)

# --- Run Button ---
st.markdown("<br>", unsafe_allow_html=True) # Add some space
col_run_button, _ = st.columns([1, 3]) # Adjust column ratio as needed
with col_run_button:
    run_button = st.button("🚀 Generate Content", type="primary", use_container_width=True)

# --- Output Section ---
st.markdown("---")
st.header("2. Generated Content")

if run_button:
    if not user_query:
        st.warning("Please enter a topic description above.")
    else:
        # Placeholder for progress logs
        log_placeholder = st.empty()
        log_messages = []

        # Define the callback function for logging
        def streamlit_callback(message):
            log_messages.append(str(message).strip())
            log_placeholder.text_area("Pipeline Progress:", "\n".join(log_messages), height=300) # Removed key="log_area"

        with st.spinner("🤖 The AI agents are working... This may take several minutes..."):
            try:
                # Define output file paths relative to the project root
                blog_md_path = os.path.join(project_root, "blog.md")
                blog_html_path = os.path.join(project_root, "blog.html")
                fact_check_path = os.path.join(project_root, "fact_check_report.md")
                streamlit_callback("Starting pipeline execution...") # Initial message
                # Run the pipeline
                # Pass the callback function to the pipeline
                blog_post_content, fact_check_report = run_pipeline(user_query, callback=streamlit_callback)

                if blog_post_content:
                    st.success("✅ Content generation complete!")

                    # Display the blog post preview (rendered from Markdown)
                    st.subheader("📄 Blog Post Preview (Rendered from Markdown)")
                    # clean_code_blocks might be useful if the AI includes ``` sometimes
                    # --- Revert to using components.html as the content is HTML ---
                    html_content_str = str(blog_post_content)

                    # Inject some basic CSS into the HTML string for better preview styling
                    # (This assumes the HTML has a <head> tag)
                    basic_styles = """
<style> /* Styles for dark background */
  body { font-family: sans-serif; color: #e0e0e0; line-height: 1.6; padding: 15px; background-color: inherit; } /* Light gray text, inherit background */
  h1, h2, h3, h4, h5, h6 { color: #ffffff; margin-top: 1.5em; margin-bottom: 0.5em; } /* White headings */
  a { color: #007bff; text-decoration: none; }
  a:hover { color: #0056b3; text-decoration: underline; }
  p { margin-bottom: 1em; }
  ul, ol { margin-bottom: 1em; padding-left: 2em; }
  li { margin-bottom: 0.5em; }
</style>
"""
                    # Use st.markdown to render the Markdown content correctly
                    st.markdown(clean_code_blocks(str(blog_post_content)), unsafe_allow_html=True) # Allow HTML in Markdown if needed for links

                    # Provide download buttons
                    st.subheader("💾 Download Files")
                    col1, col2, col3 = st.columns(3)
                    # Read files *after* pipeline confirms they should exist
                    if os.path.exists(blog_md_path):
                        with open(blog_md_path, "r", encoding="utf-8") as f:
                            col1.download_button("Download Markdown (.md)", f.read(), "blog.md", "text/markdown", use_container_width=True)
                    if os.path.exists(blog_html_path):
                        with open(blog_html_path, "r", encoding="utf-8") as f:
                            col2.download_button("Download HTML (.html)", f.read(), "blog.html", "text/html", use_container_width=True)
                    if fact_check_report and isinstance(fact_check_report, str) and os.path.exists(fact_check_path):
                        st.markdown("---")
                        st.subheader("🧐 Fact-Check Report")
                        st.markdown(fact_check_report)
                        with open(fact_check_path, "r", encoding="utf-8") as f:
                            col3.download_button("Download Fact-Check Report (.md)", f.read(), "fact_check_report.md", "text/markdown", use_container_width=True)

                else:
                    st.error("😔 Content generation failed or returned empty. Please check the console logs for errors.")

            except Exception as e:
                st.error(f"An error occurred during the pipeline execution:")
                st.error(traceback.format_exc()) # Show detailed error in the app
else:
    st.info("Enter a topic and click 'Generate Content' to start.")