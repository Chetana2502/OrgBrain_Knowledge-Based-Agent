import os
from typing import Dict, Any, List
from llama_index.core import VectorStoreIndex
from groq import Groq
from .prompts import build_system_prompt

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Central model name – change in one place if needed
GROQ_MODEL = "llama-3.3-70b-versatile"


def rewrite_query(question: str) -> str:
    """Use Groq to rewrite the user query for better retrieval."""
    prompt = f"""
Rewrite the following user query to be clearer and more specific for document search.
Return only the rewritten query text.

Original query:
{question}
"""
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return completion.choices[0].message.content.strip()


def generate_followups(answer_text: str, original_question: str) -> List[str]:
    """Generate 3 follow-up questions using Groq."""
    prompt = f"""
You are helping generate follow-up questions for a Q&A agent.

Original question: {original_question}
Answer: {answer_text}

Suggest 3 short follow-up questions the user might ask next.
Return them as a numbered list.
"""
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    text = completion.choices[0].message.content
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        # Remove leading numbers or bullets
        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        line = line.lstrip("-• ").strip()
        if line:
            cleaned.append(line)
    return cleaned


def compute_confidence(scores: List[float]) -> str:
    """Compute a simple High/Medium/Low confidence based on similarity scores."""
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg >= 0.8:
        return "High"
    elif avg >= 0.6:
        return "Medium"
    else:
        return "Low"


def answer_question(
    index: VectorStoreIndex,
    question: str,
    mode: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Main RAG pipeline:
    - Build system prompt for selected mode
    - Rewrite user query
    - Retrieve top-k chunks from index
    - Use Groq model to generate grounded answer
    - Compute confidence + follow-ups
    """
    system_prompt = build_system_prompt(mode)
    rewritten = rewrite_query(question)

    # Retrieve from LlamaIndex (top 4 chunks)
    query_engine = index.as_query_engine(similarity_top_k=4, response_mode="no_text")
    response = query_engine.query(rewritten)

    sources_info: List[Dict[str, Any]] = []
    scores: List[float] = []

    for node_with_score in getattr(response, "source_nodes", []):
        node = node_with_score.node
        score = node_with_score.score
        meta = node.metadata or {}
        doc_id = meta.get("file_name") or meta.get("source") or "Unknown document"

        text = getattr(node, "text", "") if not isinstance(node, dict) else node.get("text", "")

        sources_info.append({
            "text": text,
            "score": score,
            "doc_id": doc_id,
        })
        if score is not None:
            scores.append(score)

    # Build context from retrieved chunks
    context_chunks = "\n\n---\n\n".join(
        f"[{s['doc_id']}, score={s['score']}] {s['text']}" for s in sources_info
    )

    final_prompt = f"""
{system_prompt}

User question:
{question}

Rewritten query:
{rewritten}

Relevant document excerpts:
{context_chunks}

Instructions:
- Answer the user's question using ONLY the excerpts above.
- If the answer is unclear or missing, say you are unsure and suggest contacting a human.
- End with a "Sources:" section listing the document names you used.
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": final_prompt}
        ],
    )

    answer = completion.choices[0].message.content.strip()
    confidence = compute_confidence(scores)
    followups = generate_followups(answer, question)

    result: Dict[str, Any] = {
        "answer": answer,
        "rewritten_query": rewritten,
        "sources": sources_info,
        "confidence": confidence,
        "followups": followups,
    }

    if debug:
        result["debug_chunks"] = sources_info
    else:
        result["debug_chunks"] = None

    return result