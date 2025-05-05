import os
import streamlit as st
import sys
import traceback
import streamlit.components.v1 as components
import streamlit as st # Ensure streamlit is imported as st
import markdown # Import the markdown library

# Ensure the main script's directory is in the path to find modules
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the main pipeline function AFTER setting the path
try:
    # Assuming main.py is in the same directory as this streamlit_app.py
    from main import run_pipeline, clean_code_blocks, get_trending_topics, search_tool # Import new function and search_tool
except ImportError as e:
    st.error(f"Error importing functions from main.py: {e}")
    st.error(f"Project Root: {project_root}")
    st.error(f"Sys Path: {sys.path}")
    st.stop()

# --- Page Config ---
st.set_page_config(layout="wide", page_title="AI Content Studio", page_icon="🚀")

# --- Custom CSS (Optional - for finer control if needed later) ---
# st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

# --- Header ---
# Center the title and description using HTML/CSS within markdown
st.markdown("""
<div style="text-align: center;">
    <h1>🚀 AI-Powered Content Studio 🚀</h1>
    <p>Welcome to the AI Content Studio! Choose a trending topic, enter your own, select your desired content format, and let the AI agents craft it for you.</p>
</div>
""", unsafe_allow_html=True)
# st.markdown("""
# Welcome to the AI Content Studio! Choose a trending topic, enter your own, select your desired content format, and let the AI agents craft it for you.
# """) # Original description markdown commented out
st.markdown("""
<style>
    /* Add some padding below the title and description */
    .stApp > header {
        margin-bottom: 20px;
    }
    /* Style the main block */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style the buttons */
    .stButton>button {
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    /* Style the radio buttons */
    .stRadio > div {
        flex-direction: row;
        gap: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Input Section ---
st.header("Configure Your Content")
st.divider()

# Use columns to center the main input elements
left_spacer, main_col, right_spacer = st.columns([1, 2, 1]) # Adjust ratios as needed (e.g., [1,3,1])

with main_col:
    # --- Trending Topics Section ---
    @st.cache_data(ttl=3600) # Cache for 1 hour
    def fetch_trends():
        print("Attempting to fetch trending topics...") # Add print statement for debugging
        try:
            topics = get_trending_topics(search_tool, num_topics=6)
            print(f"Fetched topics: {topics}") # Debug print
            return topics
        except Exception as e:
            print(f"Error in fetch_trends: {e}") # Debug print
            return []

    trending_topics = fetch_trends()

    if trending_topics:
        st.subheader("🔥 Trending Topics (Click to select)")
        cols = st.columns(3) # Adjust number of columns as needed
        # Initialize session state for text area if it doesn't exist
        if 'user_query_input' not in st.session_state:
            st.session_state.user_query_input = ""

        for i, topic in enumerate(trending_topics):
            with cols[i % 3]:
                # Wrap button in a container with a border for visual separation
                container = st.container(border=True)
                if container.button(topic, key=f"topic_{i}", use_container_width=True):
                    st.session_state.user_query_input = topic  # Update session state on button click
                    st.rerun() # Rerun to reflect the change in the text area immediately
                # Add a small vertical space below each container
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) # Add space after topics

    # --- Topic Input ---
    st.text_area(
        "**Or enter your topic description below:**",
        height=100,
        placeholder="e.g., The future of renewable energy in India, or Explain the basics of quantum computing",
        key='user_query_input'
    )
    st.markdown("<br>", unsafe_allow_html=True) # Add space

    # --- Content Type Selection ---
    st.markdown("**Select Content Type:**")
    content_type_options = ["Blog Post", "Social Media Posts", "Video/Podcast Script"]
    if 'content_type' not in st.session_state:
        st.session_state.content_type = content_type_options[0] # Default to Blog Post

    st.session_state.content_type = st.radio(
        "Select the type of content to generate:", # Label hidden by markdown above
        options=content_type_options,
        index=content_type_options.index(st.session_state.content_type), # Keep selection sticky
        horizontal=True,
        label_visibility="collapsed" # Hide the default radio label
    )

    # --- Conditional Input for Script Length ---
    if st.session_state.content_type == "Video/Podcast Script":
        # Initialize if not present
        if 'script_length_minutes' not in st.session_state:
            st.session_state.script_length_minutes = 5 # Default length
        st.number_input(
            "Approximate Script Length (minutes):",
            min_value=1, max_value=60,
            key='script_length_minutes', # Use key for session state
            step=1,
            help="Enter the desired duration for the video/podcast script."
        )

    st.markdown("<br>", unsafe_allow_html=True) # Add space before button

# Initialize session state for blog content if it doesn't exist
if 'blog_post_content' not in st.session_state:
    st.session_state.blog_post_content = None
if 'fact_check_report' not in st.session_state:
    st.session_state.fact_check_report = None

# --- Run Button ---
with main_col: # Place button in the center column
    run_button = st.button("🚀 Generate Content", type="primary", use_container_width=True)


# --- Output Section ---
# (This section remains outside the centering columns to use full width)

if run_button:
    if not st.session_state.get('user_query_input', ''):
        st.warning("Please enter a topic description above.")
    else:
        with main_col, st.spinner("🤖 The AI agents are working... This may take several minutes..."): # Show spinner in center column
            try:
                query_to_run = st.session_state.user_query_input
                selected_content_type = st.session_state.content_type # Get selected type
                # Pass script length if relevant
                script_length = None
                if selected_content_type == "Video/Podcast Script":
                    script_length = st.session_state.script_length_minutes

                blog_post_content, fact_check_report = run_pipeline(
                    query_to_run, content_type=selected_content_type, script_length=script_length, callback=None) # Pass content_type and script_length

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

# --- Display Area (Remains full width) ---
st.markdown("---")
st.header("Generated Content")
st.divider()

# Display content if it exists in session state
if st.session_state.get('blog_post_content'):
    # Display the blog post preview (rendered from Markdown)
    st.subheader("📄 Content Preview")
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
            file_name="generated_content.md", # Generic name
            mime="text/markdown",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error preparing Markdown download data: {e}")

    # --- Prepare HTML Download Data (Only for Blog Posts) ---
    if st.session_state.content_type == "Blog Post":
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

