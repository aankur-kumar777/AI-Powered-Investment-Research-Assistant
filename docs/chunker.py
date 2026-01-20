
"""docs/chunker.py

Utilities to chunk long text into token-aware chunks suitable for embeddings or LLM prompts.
This module uses the Hugging Face `transformers` tokenizer (e.g., GPT-2 tokenizer) so chunk sizes
are in terms of tokens rather than words.

Requirements: transformers
"""
from typing import List
from transformers import AutoTokenizer


class TokenChunker:
    def __init__(self, tokenizer_name: str = "gpt2", chunk_size: int = 800, overlap: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # ensure pad token exists for some tokenizers
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": ""})
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Return a list of text chunks where each chunk is <= chunk_size tokens and
        consecutive chunks overlap by `overlap` tokens.
        """
        if not text:
            return []

        tok = self.tokenizer.encode(text)
        chunks = []
        start = 0
        n = len(tok)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_tokens = tok[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunks.append(chunk_text)
            if end == n:
                break
            start = end - self.overlap
            if start < 0:
                start = 0
        return chunks


def simple_whitespace_chunk(text: str, chunk_size_words: int = 200, overlap_words: int = 20) -> List[str]:
    """A simpler tokenizer-free chunker based on whitespace counts. Useful where transformers isn't available."""
    if not text:
        return []
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i+chunk_size_words]))
        i += chunk_size_words - overlap_words
    return out


if __name__ == "__main__":
    sample = "This is a sample text. " * 1000
    c = TokenChunker(chunk_size=120, overlap=20)
    chunks = c.chunk_text(sample)
    print(f"Produced {len(chunks)} chunks. First chunk length (chars): {len(chunks[0])}")
