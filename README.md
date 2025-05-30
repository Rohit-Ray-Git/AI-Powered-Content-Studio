# 🚀 AI-Powered Content Studio 🚀

![screenshot](docs/screenshot.png) <!-- Replace with an actual screenshot or GIF of your app UI -->

This project implements a multi-agent system using CrewAI and Streamlit to create a versatile content generation studio. Users can generate various types of content based on a chosen topic, leveraging AI agents for planning, research, writing, editing, SEO optimization, and fact-checking.

## ✨ Features

*   **Multi-Agent Workflow:** Utilizes a pipeline of specialized AI agents (Planner, Researcher, Writer, Reviewer, Editor, SEO Specialist, Fact Checker) powered by Google Gemini via CrewAI.
*   **Streamlit UI:** Provides an interactive web interface for easy use.
*   **Trending Topics:** Suggests current trending topics in technology to inspire content creation.
*   **Multiple Content Types:** Generate different formats:
    *   **Blog Posts:** Comprehensive, engaging articles with SEO optimization.
    *   **Social Media Posts:** Concise, attractive snippets suitable for platforms like Twitter and LinkedIn.
    *   **Video/Podcast Scripts:** Formatted scripts with options for approximate length.
*   **Multi-Language Support:** Generate content in:
    *   English (Default)
    *   Hindi
    *   Odia
*   **Customizable Script Length:** Specify the approximate desired duration for generated scripts.
*   **Download Options:** Download the generated content as:
    *   Markdown (.md)
    *   HTML (.html) (for Blog Posts)
    *   Fact-Check Report (.md) (if applicable)
*   **Professional UI:** Centered layout with clear sections and styling.

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rohit-Ray-Git/AI-Powered-Content-Studio.git
    cd AI-Powered-Content-Studio-Project
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the project root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    SERPER_API_KEY=your_serper_api_key_here
    ```

## ▶️ Usage

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit application:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Open the provided URL in your web browser.
4.  Select a trending topic or enter your own.
5.  Choose the desired content type and output language.
6.  If generating a script, specify the approximate length.
7.  Click "Generate Content" and wait for the AI agents to complete the task.
8.  Preview the generated content and use the download buttons.

## 🎬 Demo

- **Blog Post Example:**
  ![blog-demo](docs/blog_demo.png) <!-- Replace with a real screenshot or GIF -->
- **Social Media Post Example:**
  ![social-demo](docs/social_demo.png)

## 🧪 Testing

To run all tests:
```bash
python -m unittest discover tests
```
To run a specific test file:
```bash
python -m unittest tests/test_main.py
```

## 🐞 Troubleshooting

- **API Key Errors:** Ensure your `.env` file is present and contains valid keys.
- **Rate Limits:** If you see rate limit errors, wait a few minutes and try again.
- **Streamlit Not Found:** Install dependencies with `pip install -r requirements.txt`.
- **Other Issues:** Check the logs in `app.log` for more details.

## 📂 File Structure (Simplified)

*   `streamlit_app.py`: The main Streamlit user interface application.
*   `main.py`: Contains the core pipeline logic (`run_pipeline`), agent setup (`build_agents`), prompt generation (`get_prompt`), and utility functions.
*   `agents/`: Directory containing individual agent class definitions (e.g., `writer_agent.py`).
*   `.env`: Stores API keys (you need to create this).
*   `requirements.txt`: Lists project dependencies.
*   `README.md`: This file.
*   `ui_feedback.py`: Utility for user feedback in the UI.
*   `tests/`: Unit and integration tests.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT License](LICENSE) <!-- Or your chosen license -->
