from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List
import os
import docx
import PyPDF2

# Load model
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
summarizer_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

# Summary length settings
SUMMARY_LENGTHS = {
    "Small": {"max_length": 60, "min_length": 20},
    "Medium": {"max_length": 120, "min_length": 40},
    "Large": {"max_length": 200, "min_length": 80},
}

def chunk_text(text: str, max_tokens: int = 1024, overlap: int = 100) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(input_ids), max_tokens - overlap):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def summarize_document(text: str, size: str = "Medium") -> str:
    config = SUMMARY_LENGTHS.get(size, SUMMARY_LENGTHS["Medium"])
    chunks = chunk_text(text)
    partial_summaries = [
        summarizer_pipeline(chunk, max_length=config["max_length"], min_length=config["min_length"], do_sample=False)[0]['summary_text']
        for chunk in chunks
    ]
    combined_summary = " ".join(partial_summaries)
    if len(partial_summaries) > 1:
        combined_summary = summarizer_pipeline(
            combined_summary,
            max_length=config["max_length"] + 20,
            min_length=config["min_length"],
            do_sample=False
        )[0]['summary_text']
    return combined_summary

def extract_text_from_file(file) -> str:
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".txt":
            return file.read().decode("utf-8")
        elif ext == ".docx":
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            return ""
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {ext} file: {e}")
        return ""
