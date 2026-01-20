"""llm/rag_agent.py

A simple, configurable Retrieval-Augmented Generation (RAG) agent using LangChain.

Features:
- Builds a RetrievalQA chain from a vectorstore and a chat model.
- Supports using OpenAI Chat models (requires OPENAI_API_KEY) or a placeholder local model.
- Exposes simple functions: build_rag_chain and ask_with_rag

Usage example:
    from embeddings.embedder import get_embedder, embed_text
    from embeddings.vector_store import init_chroma_client, upsert_documents
    from llm.rag_agent import build_rag_chain, ask_with_rag

    # embed & upsert small example
    m = get_embedder()
    vec = embed_text(m, 'This is a test doc about ACME corp')
    init_chroma_client('./chroma_db')
    upsert_documents('research', ids=['doc1'], documents=['ACME details'], embeddings=[vec.tolist()], metadatas=[{'source':'test'}])

    chain = build_rag_chain()
    ans = ask_with_rag(chain, 'Summarize ACME corp')
    print(ans)
"""
import os
from typing import Any, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


def build_rag_chain(chroma_persist_directory: str = './chroma_db', embedding_model_name: str = 'all-MiniLM-L6-v2', openai_api_key: Optional[str] = None, temperature: float = 0.0) -> RetrievalQA:
    """Build and return a LangChain RetrievalQA chain.

    If OPENAI_API_KEY is set (or passed), ChatOpenAI will be used. Otherwise, LangChain's local fallback is attempted (may require additional setup).
    """
    # embeddings adapter for langchain
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # initialize chroma vectorstore wrapper
    vect = Chroma(persist_directory=chroma_persist_directory, embedding_function=embeddings)

    # choose LLM
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if api_key:
        model = ChatOpenAI(temperature=temperature, openai_api_key=api_key)
    else:
        # This will raise if no local LLM is configured in LangChain. User should set OPENAI_API_KEY for easy start.
        model = ChatOpenAI(temperature=temperature)

    chain = RetrievalQA.from_chain_type(llm=model, chain_type='map_reduce', retriever=vect.as_retriever(search_kwargs={'k': 5}))
    return chain


def ask_with_rag(chain: RetrievalQA, question: str) -> Any:
    """Ask a question through the RAG chain and return the answer object (may contain text and metadata depending on chain)."""
    return chain.run(question)