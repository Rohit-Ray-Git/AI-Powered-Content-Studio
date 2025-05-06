from main import run_pipeline, clean_code_blocks
import markdown

if __name__ == "__main__":
    user_query = "Write a detailed report on the impact of AI in human life."
    print(f"\nRunning multi-agent pipeline via demo script for query: {user_query}\n")
    # Pass print function as a simple callback when running directly
    final_content, fact_check_report = run_pipeline(user_query, content_type="Blog Post", language="English", tone="Informational", callback=print) # Specify tone
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
    full_html = f"""<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\"><title>AI Generated Blog Post</title></head>\n<body>{html_content}</body>\n</html>"""

    with open("blog.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    print("\nBlog article converted to HTML and saved to blog.html.\n")
    if fact_check_report and isinstance(fact_check_report, str):
        with open("fact_check_report.md", "w", encoding="utf-8") as f:
            f.write(fact_check_report)
        print("\nFact-check report saved to fact_check_report.md.\n") 