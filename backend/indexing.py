import os
from typing import Optional
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.llms.mock import MockLLM

from llama_index.core.embeddings import resolve_embed_model

# Use a local HuggingFace embedding model instead of OpenAI
# This avoids needing OPENAI_API_KEY
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en")

# Use a mock LLM so LlamaIndex doesn't require OpenAI
Settings.llm = MockLLM()

def build_index_from_dir(doc_dir: str | Path):
    """Build a vector index from all documents in a directory."""
    doc_dir = Path(doc_dir)
    docs = SimpleDirectoryReader(str(doc_dir)).load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index


def build_index_from_dir(doc_dir: str) -> Optional[VectorStoreIndex]:
    """
    Build an index from all documents in the given directory.
    Returns None if no documents are found.
    """
    if not os.path.exists(doc_dir):
        return None

    files = [
        os.path.join(doc_dir, f)
        for f in os.listdir(doc_dir)
        if os.path.isfile(os.path.join(doc_dir, f))
    ]
    if not files:
        return None

    # Load all documents
    docs = SimpleDirectoryReader(doc_dir, recursive=True).load_data()
    if not docs:
        return None

    # Create in-memory vector index
    index = VectorStoreIndex.from_documents(docs)
    return index