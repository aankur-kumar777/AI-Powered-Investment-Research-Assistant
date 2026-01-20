# api/deps.py
from functools import lru_cache
from typing import Optional

from embeddings.vector_store import init_chroma_client
from llm.rag_agent import build_rag_chain

# singleton vector store client (Chroma)
@lru_cache()
def chroma_client(persist_directory: Optional[str] = "./chroma_db"):
    return init_chroma_client(persist_directory)

# singleton RAG chain
@lru_cache()
def rag_chain(openai_api_key: Optional[str] = None, chroma_persist_directory: Optional[str] = "./chroma_db"):
    # If OPENAI_API_KEY is in env, build_rag_chain will use it automatically.
    return build_rag_chain(chroma_persist_directory=chroma_persist_directory,
                           openai_api_key=openai_api_key)
