"""embeddings/vector_store.py

Lightweight Chroma wrapper for upsert & semantic search.

Functions:
- init_chroma_client(persist_directory)
- upsert_documents(collection_name, ids, documents, embeddings, metadatas)
- semantic_search(collection_name, query_embedding, top_k)
- get_collection_info(collection_name)

Note: This module uses chromadb. For production prefer Qdrant/Pinecone connectors.
"""
from typing import List, Dict, Any, Optional
from chromadb import Client
from chromadb.config import Settings
import os


_chroma_client: Optional[Client] = None


def init_chroma_client(persist_directory: Optional[str] = './chroma_db') -> Client:
    """Initialize a chroma client (singleton). If persist_directory is None, it will use an ephemeral directory './chroma_db'."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    os.makedirs(persist_directory, exist_ok=True)
    settings = Settings(chroma_db_impl='duckdb+parquet', persist_directory=persist_directory)
    _chroma_client = Client(settings)
    return _chroma_client


def upsert_documents(collection_name: str, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
    client = init_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(ids=ids, documents=documents, metadatas=metadatas or [{}]*len(ids), embeddings=embeddings)


def semantic_search(collection_name: str, query_embedding: List[float], top_k: int = 5):
    client = init_chroma_client()
    collection = client.get_collection(collection_name)
    res = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # res contains dict with keys 'ids', 'documents', 'metadatas', 'distances'
    return res


def delete_collection(collection_name: str):
    client = init_chroma_client()
    client.delete_collection(name=collection_name)


def list_collections():
    client = init_chroma_client()
    return client.list_collections()


if __name__ == '__main__':
    # quick demo
    m = init_chroma_client()
    print('Collections:', list_collections())