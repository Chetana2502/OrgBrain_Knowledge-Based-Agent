# Architecture Description

Use this to draw a nice diagram later:

```markdown
# OrgBrain Architecture

## Components

1. **User**
   - Interacts via a web browser.

2. **Frontend (Streamlit)**
   - File Uploader
   - Mode Selector (General / HR / Support / Operations)
   - Chat Interface
   - Debug Mode Toggle
   - Document Insights Dashboard

3. **Backend (Python)**
   - **Indexing Module (LlamaIndex)**
     - Reads documents from `data/uploaded`
     - Creates and holds a `VectorStoreIndex`
   - **RAG Pipeline**
     - Query Rewriter (Groq LLaMA model)
     - Retriever (top-k document chunks from the index)
     - Answer Generator (Groq LLaMA model using retrieved chunks + system prompt)
     - Confidence Scorer (based on similarity scores)
     - Follow-Up Question Generator (Groq LLaMA model)
   - **Document Insights Engine**
     - Extracts raw text from each document
     - Summarizes content
     - Extracts key points
     - Suggests relevant roles for the document

4. **External Service**
   - **Groq API**
     - Provides hosted LLaMA-based models for:
       - Query rewriting
       - Answer generation
       - Follow-up question generation
       - Document summarization

5. **Storage**
   - Local file storage (`data/uploaded`) for uploaded documents
   - In-memory vector index (inside the running app process)

## Flow

1. **Document Upload**
   - User uploads files via Streamlit.
   - Streamlit saves them into `data/uploaded`.

2. **Index Building**
   - User clicks "Rebuild Index".
   - Indexing module loads all files and builds a `VectorStoreIndex`.

3. **Question Answering**
   - User enters question and selects a mode.
   - RAG pipeline:
     - Rewrites question (Groq).
     - Retrieves top-k chunks from `VectorStoreIndex`.
     - Sends system prompt + mode + retrieved chunks to Groq model.
     - Receives grounded answer, computes confidence, and generates follow-ups.
   - Streamlit displays answer, confidence, sources, and (optionally) debug chunks.

4. **Document Insights**
   - For each document in `data/uploaded`:
     - Text is extracted (PDF/TXT).
     - Groq model generates a summary with key points and role suggestions.
   - Streamlit displays these insights in an expandable list.
