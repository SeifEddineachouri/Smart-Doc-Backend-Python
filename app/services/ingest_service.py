from app.core.config import settings


def chunk_text(content: str) -> list[str]:
    words = [word for word in content.split() if word]
    if not words:
        return []

    chunk_size = max(1, settings.chunk_size_words)
    overlap = max(0, min(settings.chunk_overlap_words, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks

