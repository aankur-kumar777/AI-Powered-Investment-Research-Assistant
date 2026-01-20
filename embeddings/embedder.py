"""embeddings/embedder.py

Wrapper around sentence-transformers with a small convenience API.

Usage:
    from embeddings.embedder import get_embedder, embed_texts
    model = get_embedder('all-MiniLM-L6-v2')
    vectors = embed_texts(model, ['hello world', 'second doc'])
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Load and return a SentenceTransformer model instance."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts into a numpy array (n_texts, dim)."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(embeddings)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """Embed a single text and return a 1-D numpy array."""
    return embed_texts(model, [text])[0]