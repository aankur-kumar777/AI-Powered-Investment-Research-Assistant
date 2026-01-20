"""tests/test_embeddings.py

Lightweight tests for embedder & vector store.
"""
import os
import shutil
from embeddings.embedder import get_embedder, embed_texts
from embeddings.vector_store import init_chroma_client, upsert_documents, semantic_search, delete_collection


def test_embedder_shape():
    model = get_embedder()
    vecs = embed_texts(model, ['hello world', 'second doc'])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] == model.get_sentence_embedding_dimension()


def test_vector_store_upsert_and_query(tmp_path):
    db_dir = tmp_path / 'chroma_test'
    init_chroma_client(str(db_dir))
    ids = ['a', 'b']
    docs = ['apple is a fruit', 'google is a company']
    model = get_embedder()
    emb = embed_texts(model, docs)
    upsert_documents('test_collection', ids=ids, documents=docs, embeddings=emb.tolist())
    q_emb = embed_texts(model, ['who is a company'])[0].tolist()
    res = semantic_search('test_collection', q_emb, top_k=2)
    assert 'documents' in res
    # cleanup
    delete_collection('test_collection')