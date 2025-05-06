import gradio as gr
from transformers import pipeline
import PyPDF2
import docx
from io import BytesIO

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text(file):
    name = file.name.lower()
    text = ""
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    else:
        return "Unsupported file type."
    return text.strip()

def split_text_into_chunks(text, max_tokens=8000):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def summarize_input(text, file):
    input_text = ""
    if file:
        input_text = extract_text(file)
    elif text.strip():
        input_text = text.strip()
    else:
        return "Please provide text or upload a file."

    if len(input_text.split()) < 30:
        return "Text too short to summarize. Please enter at least 30 words."

    chunks = split_text_into_chunks(input_text)
    summaries = [summarizer(chunk, max_length=15000, min_length=40, do_sample=False)[0]["summary_text"] for chunk in chunks]
    return " ".join(summaries)

iface = gr.Interface(
    fn=summarize_input,
    inputs=[
        gr.Textbox(label="Enter text", lines=6, placeholder="Or upload a file below..."),
        gr.File(label="Upload .pdf, .docx or .txt")
    ],
    outputs="text",
    title="SmartSummarize",
    description="Upload a document or paste text to get a summary powered by transformers."
)

if __name__ == "__main__":
    iface.launch()
