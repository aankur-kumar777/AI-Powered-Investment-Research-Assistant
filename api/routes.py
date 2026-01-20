# api/routes.py
import os
import tempfile
import uuid
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from api.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse
from api.deps import chroma_client, rag_chain

from docs.processor import extract_text_from_pdf
from docs.chunker import TokenChunker, simple_whitespace_chunk
from embeddings.embedder import get_embedder, embed_texts
from embeddings.vector_store import upsert_documents, semantic_search

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()

@router.post("/ingest", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    persist_dir: str = "./chroma_db",
    tokenizer_name: str = "gpt2",
    chunk_size: int = 800,
    overlap: int = 100,
    collection: str = "research_docs",
):
    """
    Upload a PDF (or similar). The route:
      - saves the uploaded file to a temporary path
      - extracts text (PyMuPDF / pdfplumber / OCR fallback)
      - chunks text (token-aware chunker; falls back to whitespace chunker)
      - embeds chunks and upserts into Chroma collection
    Returns number of chunks ingested.
    """
    # save uploaded file to temp
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".pdf")
    try:
        tmp.write(file.file.read())
        tmp.flush()
        tmp.close()
        # extract text
        try:
            text = extract_text_from_pdf(tmp.name)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Uploaded file not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {e}")

        # chunk text
        try:
            chunker = TokenChunker(tokenizer_name=tokenizer_name, chunk_size=chunk_size, overlap=overlap)
            chunks = chunker.chunk_text(text)
            if not chunks:
                # fallback
                chunks = simple_whitespace_chunk(text, chunk_size_words=200, overlap_words=20)
        except Exception:
            # Robust fallback
            chunks = simple_whitespace_chunk(text, chunk_size_words=200, overlap_words=20)

        # embed chunks
        embedder = get_embedder()  # default all-MiniLM-L6-v2
        embeddings = embed_texts(embedder, chunks)

        # prepare ids & metadata
        ids = []
        metadatas = []
        docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{uuid.uuid4().hex[:12]}_{i}"
            ids.append(chunk_id)
            docs.append(chunk_text)
            metadatas.append({"source_file": file.filename, "chunk_index": i})

        # upsert into chroma
        upsert_documents(collection, ids=ids, documents=docs, embeddings=embeddings.tolist(), metadatas=metadatas)

        return IngestResponse(status="ingested", ingested_chunks=len(chunks), collection=collection)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, top_k_override: int = None, persist_dir: str = "./chroma_db"):
    """
    Query the RAG chain. This endpoint:
      - runs the RetrievalQA chain (langchain) to produce an answer
      - also performs a semantic search to return sources/documents
    """
    # build rag chain and vectorstore client (singletons handled in deps)
    chain = rag_chain(chroma_persist_directory=persist_dir)
    # 1) Ask RAG for the answer
    try:
        answer = chain.run(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG chain error: {e}")

    # 2) Do a semantic search to return top sources (best-effort)
    try:
        # use same embedder to get query embedding
        embedder = get_embedder()
        q_emb = embed_texts(embedder, [req.query])[0].tolist()
        res = semantic_search("research_docs", q_emb, top_k=req.top_k)
        # extract documents or metadata (chroma response shape may vary)
        sources: List[str] = []
        if res:
            # prefer metadatas if available
            m = res.get("metadatas") or []
            docs = res.get("documents") or []
            # collect a readable source string per result
            for idx in range(len(docs)):
                md = m[idx] if idx < len(m) else {}
                src = md.get("source_file") if isinstance(md, dict) else None
                if not src:
                    src = f"doc[{idx}]"
                sources.append(src)
        else:
            sources = []
    except Exception:
        sources = []

    return QueryResponse(answer=str(answer), sources=sources)
