import os
from typing import List
from groq import Groq
from pypdf import PdfReader

# Initialize Groq client using GROQ_API_KEY from environment (.env is loaded in app.py)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def list_document_paths(doc_dir: str) -> List[str]:
    """List all file paths in the document directory."""
    if not os.path.exists(doc_dir):
        return []
    return [
        os.path.join(doc_dir, f)
        for f in os.listdir(doc_dir)
        if os.path.isfile(os.path.join(doc_dir, f))
    ]


def get_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text_chunks = []
    for page in reader.pages:
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_chunks)


def get_text_from_file(path: str) -> str:
    """Return text from PDF or TXT file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return get_text_from_pdf(path)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        # Unsupported types can be added later (e.g. .docx)
        return ""


def summarize_document(text: str, filename: str) -> str:
    """
    Summarize a document with roles and key points using a Groq model.
    """
    if not text.strip():
        return "No readable text extracted from this document."

    truncated = text[:6000]  # avoid super-long prompts

    prompt = f"""
You are summarizing an internal company document.

File name: {filename}

1. Give a 5–7 line summary.
2. List 5 key bullet points.
3. Suggest which roles should read this document (e.g., HR, Support, Managers, Engineers).

Document text:
{truncated}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # adjust if Groq updates model names
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return completion.choices[0].message.content.strip()
