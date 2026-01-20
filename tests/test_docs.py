
"""tests/test_docs.py

Tests for docs/processor.py and docs/chunker.py. These tests are lightweight and do not need PDFs.
They validate that chunking works and that processor saves a sample text file when given a small sample.
"""
import os
import tempfile
from docs import chunker
from docs import processor


def test_whitespace_chunker():
    text = "word " * 1000
    chunks = chunker.simple_whitespace_chunk(text, chunk_size_words=200, overlap_words=20)
    assert len(chunks) > 0
    # check overlap by verifying that subsequent chunk shares words
    first = chunks[0].split()
    second = chunks[1].split()
    assert first[-20:] == second[:20]


def test_token_chunker_small():
    text = "This is a simple sentence. " * 200
    tc = chunker.TokenChunker(chunk_size=100, overlap=10)
    chunks = tc.chunk_text(text)
    assert len(chunks) > 0


def test_save_extracted_text_tmpfile():
    # we can't rely on a PDF in tests; instead write a tiny PDF-like placeholder using plain text
    tmp_dir = tempfile.mkdtemp()
    out_txt = os.path.join(tmp_dir, "out.txt")
    # simulate `extract_text_from_pdf` by monkeypatching it
    original = processor.extract_text_from_pdf
    try:
        processor.extract_text_from_pdf = lambda p: "Hello world"
        result = processor.save_extracted_text("dummy.pdf", out_path=out_txt)
        assert os.path.exists(result)
        with open(result, "r", encoding="utf-8") as f:
            assert f.read() == "Hello world"
    finally:
        processor.extract_text_from_pdf = original
