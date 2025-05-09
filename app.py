import logging
import gradio as gr
from summarizer import summarize_document, extract_text_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

def handle_summary(text: str, size: str, file) -> str:
    try:
        content = text.strip()
        if file:
            content = extract_text_from_file(file)
            if not content:
                return "❌ Could not extract text from the file. Please try another file."

        if not content:
            return "❗ Please enter text or upload a valid file."

        summary = summarize_document(content, size)
        return summary
    except Exception as e:
        logging.exception("Error during summarization")
        return f"❌ Error: {e}"

def launch_ui():
    with gr.Blocks(title="Smart Summarizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 Smart Document Summarizer")
        gr.Markdown("Upload a `.pdf`, `.docx`, or `.txt` file or paste your text, then choose summary length.")

        with gr.Row():
            input_text = gr.Textbox(label="📝 Paste Text Here", lines=15, placeholder="Or upload a file below...")
        
        with gr.Row():
            file_upload = gr.File(label="📎 Upload File", file_types=[".pdf", ".docx", ".txt"])
        
        with gr.Row():
            size_option = gr.Radio(choices=["Small", "Medium", "Large"], label="🧠 Summary Length", value="Medium")
        
        with gr.Row():
            output_text = gr.Textbox(label="📃 Summary", lines=10)

        summarize_btn = gr.Button("✨ Generate Summary")
        summarize_btn.click(fn=handle_summary, inputs=[input_text, size_option, file_upload], outputs=output_text)

    app.launch()

if __name__ == "__main__":
    launch_ui()
