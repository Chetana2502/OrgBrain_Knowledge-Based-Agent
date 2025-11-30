# OrgBrain – Adaptive Knowledge Base AI Agent

OrgBrain is a real-time, Groq-powered AI agent that helps organizations instantly retrieve answers from internal documents such as HR policies, SOPs, FAQs, guidelines, or operational manuals. Users can upload PDF/TXT files, and the system builds a searchable knowledge base using RAG (Retrieval-Augmented Generation) to answer questions with citations, confidence scores, and document-derived context.

---

## 1. Overview

Employees often struggle to find information buried inside long documents. OrgBrain solves this by allowing teams to upload documents, automatically index them, and query them using natural language.

OrgBrain provides:
- Accurate answers based only on company documents.
- Transparent retrieval with document citations.
- Adaptive answering behavior based on selected modes (HR, Support, Operations, General).
- Instant insights for each uploaded document.

This agent is designed to be practical, deployable, and ready for real organizational use.

---

## 2. Features

### Multi-Mode AI
Supports four intelligent modes:
- General
- HR
- Support
- Operations

Each mode adjusts tone and response focus.

### Real-Time RAG System
- Uses LlamaIndex to index uploaded documents.
- Retrieves top-k relevant chunks.
- Uses Groq-hosted LLaMA models for fast and grounded answers.

### Query Rewriting
Automatically rewrites user questions for improved document search accuracy.

### Confidence Scoring
Outputs High/Medium/Low confidence based on similarity scores of retrieved text.

### Citations and Transparency
Every answer includes:
- Sources used
- Similarity scores
- Retrieved chunks (in debug mode)

### Suggested Follow-Up Questions
Provides 3 next-step questions after every answer to help users explore further.

### Document Insights Dashboard
Automatically generates:
- Summaries
- Key bullet points
- Recommended roles for each document

---

## 3. Tech Stack

- Language: Python 3.10+
- UI: Streamlit
- LLM Backend: Groq API (LLaMA 3.3 models)
- RAG Framework: LlamaIndex
- Vector Store: In-memory VectorStoreIndex
- Libraries: groq, pypdf, python-dotenv, chromadb

---

## 4. Architecture

### Components

1. **User Interface (Streamlit)**
   - Document upload
   - Chat interface
   - Mode selection
   - Debug mode
   - Document insights page

2. **Indexing Layer**
   - Uses LlamaIndex to process and embed documents.
   - Creates a VectorStoreIndex stored in memory.

3. **RAG Pipeline**
   - Query rewriting (Groq)
   - Retrieval of relevant chunks
   - Answer generation using Groq LLaMA models
   - Confidence scoring
   - Suggested follow-up questions

4. **Document Insights Engine**
   - Extracts text from PDFs/TXT
   - Summarizes content
   - Generates key points and suggests audience roles

5. **Groq API**
   - Provides extremely fast inference.
   - Handles all LLM tasks (rewriting, summarization, answering).

---

## 5. Folder Structure

```
orgbrain/
│── app.py
│── README.md
│── requirements.txt
│── architecture.md
│── .env
│
├── backend/
│   ├── indexing.py
│   ├── rag_pipeline.py
│   ├── prompts.py
│   ├── utils.py
│   └── __init__.py
│
└── data/
    └── uploaded/
```

---

## 6. Setup Instructions

### 1. Clone the repository
```
git clone https://github.com/your-username/orgbrain.git
cd orgbrain
```

### 2. Create and activate virtual environment
```
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Add Groq API key
Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 7. Running the Application

```
streamlit run app.py
```

Open the browser at:
```
http://localhost:8501
```

Upload documents, rebuild index, choose a mode, and start asking questions.

---

## 8. Usage Instructions

1. Upload PDF/TXT documents from the sidebar.
2. Click "Rebuild Index" to process them.
3. Select a mode (General, HR, Support, Operations).
4. Ask a natural language question.
5. Review:
   - Answer
   - Citations
   - Confidence score
   - Suggested follow-up questions
6. Enable Debug Mode to view retrieved chunks.
7. Visit the Document Insights tab to view summaries.

---

## 9. Limitations

- Currently supports PDF and TXT formats.
- In-memory vector storage resets on app restart.
- Requires manual index rebuilding after uploading documents.
- No authentication system.
- Works best for English content.

---

## 10. Future Improvements

- Add support for DOCX and HTML.
- Persistent vector store (Pinecone/Chroma).
- Multi-user authentication and access control.
- Real-time analytics dashboard.
- Integration with HRMS, ticketing, or internal tools.
- REST API endpoints for external integration.

---

## 11. Credits

Built as part of the AI Agent Development Challenge — demonstrating real-world AI engineering by combining RAG, real-time Groq inference, and adaptive conversational reasoning.

