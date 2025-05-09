import logging
import gradio as gr
from summarizer import summarize_document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

def handle_summary(text: str, size: str) -> str:
    """
    Handles summarization based on user input.
    
    Args:
        text (str): Full input document text.
        size (str): Summary size - Small, Medium, Large.
    
    Returns:
        str: Generated summary or error message.
    """
    try:
        if not text.strip():
            return "❗ Please enter some text to summarize."

        summary = summarize_document(text, size)
        logging.info(f"Generated a {size.lower()} summary.")
        return summary

    except Exception as e:
        logging.exception("Error while generating summary:")
        return f"❌ An unexpected error occurred: {str(e)}"

def launch_ui():
    """Launches the Gradio interface."""
    with gr.Blocks(title="Smart Summarizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧠 Smart Document Summarizer")
        gr.Markdown("Paste a large document below and select the desired summary length.")

        input_text = gr.Textbox(lines=20, label="📄 Input Document")
        size_option = gr.Radio(choices=["Small", "Medium", "Large"], value="Medium", label="📝 Summary Length")
        output_text = gr.Textbox(lines=10, label="📃 Summary")

        summarize_btn = gr.Button("✨ Generate Summary")
        summarize_btn.click(fn=handle_summary, inputs=[input_text, size_option], outputs=output_text)

    app.launch()

if __name__ == "__main__":
    launch_ui()
