from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import os
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text_from_file(file):
    text = ""
    filename = file.filename.lower()
    if filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def split_text_into_chunks(text, max_tokens=800):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = ""
    if request.is_json:
        data = request.get_json()
        input_text = data.get("text", "").strip()
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        try:
            input_text = extract_text_from_file(file)
        except Exception as e:
            return jsonify({"error": f"File extraction failed: {str(e)}"}), 500
    else:
        return jsonify({"error": "No valid input provided."}), 400

    if not input_text or len(input_text.split()) < 30:
        return jsonify({"error": "Text too short to summarize. Please enter at least 30 words."}), 400

    chunks = list(split_text_into_chunks(input_text))
    summaries = [
        summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        for chunk in chunks
    ]
    return jsonify({"summary": " ".join(summaries)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

