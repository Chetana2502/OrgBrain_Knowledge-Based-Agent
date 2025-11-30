import os
import sys
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

st.write("Loading!!", os.environ.get("GROQ_API_KEY") is not None)

# Project root (folder containing app.py and backend/)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from the backend packages
from backend.indexing import build_index_from_dir
from backend.rag_pipeline import answer_question
from backend.utils import (
    list_document_paths,
    get_text_from_file,
    summarize_document,
)

# Where uploaded documents are stored
DOC_DIR = ROOT / "data" / "uploaded"
os.makedirs(DOC_DIR, exist_ok=True)

# Load environment variables from .env (for GROQ_API_KEY)


DOC_DIR = ROOT / "data" / "uploaded"
os.makedirs(DOC_DIR, exist_ok=True)

st.set_page_config(
    page_title="OrgBrain – Adaptive Knowledge Base Agent",
    layout="wide",
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index" not in st.session_state:
    st.session_state.index = build_index_from_dir(DOC_DIR)

if "doc_insights" not in st.session_state:
    st.session_state.doc_insights = {}  # filename -> summary text


# ---------------- SIDEBAR ----------------
st.sidebar.title("OrgBrain Controls")

mode = st.sidebar.selectbox(
    "Agent Mode",
    options=["General", "HR", "Support", "Operations"],
    help="Mode slightly changes how the agent responds and what it focuses on.",
)

debug = st.sidebar.checkbox(
    "Debug Mode (show retrieved chunks)",
    value=False,
    help="Shows raw chunks and similarity scores used to answer.",
)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload company policies, FAQs, SOPs, etc.",
)

if uploaded_files:
    for f in uploaded_files:
        save_path = os.path.join(DOC_DIR, f.name)
        with open(save_path, "wb") as out:
            out.write(f.read())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s). Click 'Rebuild Index' to include them.")

if st.sidebar.button("Rebuild Index"):
    with st.spinner("Building index from documents..."):
        st.session_state.index = build_index_from_dir(DOC_DIR)
    if st.session_state.index is None:
        st.sidebar.error("No valid documents found. Please upload PDFs or TXT files.")
    else:
        st.sidebar.success("Index built successfully!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Confidence Legend**")
st.sidebar.markdown("- High: reliable\n- Medium: okay\n- Low: verify with a human")


# ---------------- MAIN TABS ----------------
tab_chat, tab_insights, tab_how = st.tabs(
    ["💬 Chat", "📄 Document Insights", "🔍 How it Works"]
)


# -------- CHAT TAB --------
with tab_chat:
    st.title("OrgBrain – Adaptive Knowledge Base Agent")
    st.caption(f"Ask anything about your uploaded documents. Current mode: **{mode}**")

    if st.session_state.index is None:
        st.info("No index available. Please upload documents in the sidebar and click 'Rebuild Index'.")
    else:
        user_question = st.text_input("Ask a question about your documents:")

        if st.button("Ask") and user_question.strip():
            with st.spinner("Thinking..."):
                result = answer_question(
                    index=st.session_state.index,
                    question=user_question,
                    mode=mode,
                    debug=debug,
                )
            st.session_state.chat_history.append(
                {"question": user_question, "result": result}
            )

        # Show chat history (latest first)
        for item in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {item['question']}")
            st.markdown(f"**OrgBrain:** {item['result']['answer']}")
            st.markdown(f"**Confidence:** {item['result']['confidence']}")
            st.markdown(f"**Interpreted as:** `{item['result']['rewritten_query']}`")

            # Follow-up questions
            if item["result"]["followups"]:
                st.markdown("**Suggested follow-up questions:**")
                for q in item["result"]["followups"]:
                    st.markdown(f"- {q}")

            # Sources
            st.markdown("**Sources used:**")
            if item["result"]["sources"]:
                for s in item["result"]["sources"]:
                    score_str = f"{s['score']:.2f}" if s["score"] is not None else "N/A"
                    st.markdown(f"- `{s['doc_id']}` (similarity score: {score_str})")
            else:
                st.markdown("- No sources (answer may be uncertain)")

            # Debug chunks
            if debug and item["result"]["debug_chunks"]:
                with st.expander("Show retrieved chunks (debug)"):
                    for i, chunk in enumerate(item["result"]["debug_chunks"], start=1):
                        score_str = f"{chunk['score']:.2f}" if chunk["score"] is not None else "N/A"
                        st.markdown(f"**Chunk {i} – {chunk['doc_id']} (score {score_str})**")
                        st.write(chunk["text"])

            st.markdown("---")


# -------- DOCUMENT INSIGHTS TAB --------
with tab_insights:
    st.header("Document Insights")

    st.write(
        "This section automatically generates concise summaries of each uploaded document, "
        "key bullet points, and suggests which roles should read it."
    )

    doc_paths = list_document_paths(DOC_DIR)
    if not doc_paths:
        st.info("No documents found. Upload files from the sidebar.")
    else:
        # Generate insights only for new documents
        for path in doc_paths:
            fname = os.path.basename(path)
            if fname not in st.session_state.doc_insights:
                with st.spinner(f"Analyzing {fname}..."):
                    text = get_text_from_file(path)
                    summary = summarize_document(text, fname)
                    st.session_state.doc_insights[fname] = summary

        # Display insights
        for fname, summary in st.session_state.doc_insights.items():
            with st.expander(fname, expanded=False):
                st.markdown(summary)


# -------- HOW IT WORKS TAB --------
with tab_how:
    st.header("How OrgBrain Works")

    st.markdown("""
### 1. Document Upload & Indexing
- You upload internal documents (e.g., HR policies, FAQs, SOPs) as PDF or TXT.
- OrgBrain reads and converts them into embeddings using a vector index (LlamaIndex).

### 2. Asking a Question
- You type a question in natural language.
- The agent first **rewrites** your question into a clearer query for document search.

### 3. Retrieval-Augmented Generation (RAG)
- The rewritten query is used to retrieve the **most relevant chunks** from the indexed documents.
- These chunks, along with a mode-specific system prompt (HR / Support / Operations / General), are sent to a Groq-hosted LLaMA model.
- The model generates an answer that is grounded in the retrieved text.

### 4. Transparency & Trust
- OrgBrain computes a **confidence score** based on similarity scores of retrieved chunks.
- The answer always includes a **Sources** section that lists which documents were used.
- In **Debug Mode**, you can see the exact retrieved chunks and their similarity scores.

### 5. Insights for Managers
- The **Document Insights** tab summarizes each document, extracts key bullet points, and suggests who should read it.
- This makes it easy for managers & new employees to understand what matters in each document quickly.
""")