"""# Chunking utilities (LAB-2)
# - Token-based chunking (uses tiktoken if available, falls back to whitespace)
# - Sentence-based chunking (simple punctuation splitter)
# - Character-based chunking
#
# Usage examples:
#   chunks = chunk_text_tokenwise(text, max_tokens=512, overlap=64)
#   chunks = chunk_text_sentencewise(text, max_chars=2000, overlap_chars=200)
#   chunks = chunk_text_charwise(text, max_chars=2000, overlap_chars=200)
#
from typing import List, Optional
import re

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def _try_get_tokenizer(tokenizer_name: str = "gpt2"):
    """
    Try to load tiktoken and return an encoder. Returns None if unavailable.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        return None

def chunk_text_tokenwise(text: str, max_tokens: int = 500, overlap: int = 50, tokenizer_name: str = "gpt2") -> List[str]:
    """
    Chunk text into token-sized pieces.
    - max_tokens: maximum tokens per chunk (must be > 0)
    - overlap: number of tokens to overlap between consecutive chunks (>= 0)
    If tiktoken is available it is used, otherwise this falls back to whitespace tokens.
    Returns list of text chunks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    encoder = _try_get_tokenizer(tokenizer_name)
    if encoder is not None:
        tokens = encoder.encode(text)
        chunks: List[str] = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + max_tokens, n)
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
            if end == n:
                break
            # advance with overlap, ensure progress
            start = max(end - overlap, end - max_tokens + 1)
        return chunks
    else:
        words = text.split()
        chunks = []
        start = 0
        n = len(words)
        while start < n:
            end = min(start + max_tokens, n)
            chunks.append(" ".join(words[start:end]))
            if end == n:
                break
            start = max(end - overlap, end - max_tokens + 1)
        return chunks

def chunk_text_sentencewise(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> List[str]:
    """
    Chunk text by sentences, trying to keep each chunk under max_chars and overlapping by approx overlap_chars.
    This is a simple splitter (based on punctuation) and not suitable for all languages or edge cases.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    chunks: List[str] = []
    current = ""
    i = 0
    while i < len(sentences):
        s = sentences[i]
        if (len(current) + (1 if current else 0) + len(s)) <= max_chars or not current:
            current = (current + " " + s).strip() if current else s
            i += 1
        else:
            chunks.append(current)
            # prepare overlap by keeping last overlap_chars characters
            if overlap_chars > 0:
                current = current[-overlap_chars:]
            else:
                current = ""
    if current:
        chunks.append(current)
    return chunks

def chunk_text_charwise(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> List[str]:
    """
    Chunk text by characters with overlap.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap_chars, end - max_chars + 1)
    return chunks

if __name__ == "__main__":
    sample = (
        "This is a short example. It demonstrates chunking by sentences, "
        "characters, or tokens. Use token-based chunking for embeddings "
        "and character/sentence chunking for simpler pipelines."
    )
    print("Sentence chunks:", chunk_text_sentencewise(sample, max_chars=50, overlap_chars=10))
    print("Char chunks:", chunk_text_charwise(sample, max_chars=50, overlap_chars=10))
    print("Token chunks (approx):", chunk_text_tokenwise(sample, max_tokens=10, overlap=2))
"""