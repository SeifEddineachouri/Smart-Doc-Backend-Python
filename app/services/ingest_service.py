import re

from app.core.config import settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def chunk_text(content: str) -> list[str]:
    """Sentence-aware chunking with word-count targets and overlap.

    Keeps sentences intact where possible so a clause, value, or obligation is
    not split across chunk boundaries, which improves both embedding quality and
    the model's ability to ground an answer in a single retrieved excerpt.
    """
    text = content.strip()
    if not text:
        return []

    chunk_size = max(1, settings.chunk_size_words)
    overlap = max(0, min(settings.chunk_overlap_words, chunk_size - 1))

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return _word_window_chunks(text.split(), chunk_size, overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        # A single oversized sentence is split on the word window directly.
        if len(sentence_words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.extend(_word_window_chunks(sentence_words, chunk_size, overlap))
            continue

        if current_words + len(sentence_words) > chunk_size and current:
            chunks.append(" ".join(current))
            # Carry the tail of the previous chunk forward for context overlap.
            tail = " ".join(current).split()[-overlap:] if overlap else []
            current = tail + sentence_words
            current_words = len(current)
        else:
            current.extend(sentence_words)
            current_words += len(sentence_words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _word_window_chunks(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    words = [word for word in words if word]
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks

