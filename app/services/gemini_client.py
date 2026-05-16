from google import genai
import re

from app.core.config import settings


class GeminiClient:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        language = _detect_language(question)

        if not contexts:
            return _no_context_message(language)

        if self._client is None:
            return _context_only_fallback(language, question, contexts)

        prompt = _build_prompt(question, contexts)

        try:
            response = self._client.models.generate_content(model=settings.model_name, contents=prompt)
            text = (response.text or "").strip()
            if not text or _looks_generic(text):
                return _context_only_fallback(language, question, contexts)
            return text
        except Exception:
            # Keep API responsive even when provider returns transient errors.
            return _context_only_fallback(language, question, contexts)


def _build_prompt(question: str, contexts: list[str]) -> str:
    formatted_contexts = "\n".join(f"- {context}" for context in contexts)
    return (
        "You are a precise document question-answering assistant.\n"
        "Use only the provided document excerpts. Do not invent facts.\n"
        "Answer the exact question, not a generic summary.\n"
        "Be concrete and specific: mention the clause, fact, value, date, condition, or obligation that directly answers the question.\n"
        "If the excerpts do not contain the answer, say exactly that the document excerpts do not contain enough information to answer this question.\n"
        "Keep it concise (2-4 sentences max).\n"
        "Cite the relevant evidence by repeating the excerpt label such as [document_id=... chunk_index=...].\n"
        "Do not mention that you are an AI model or talk about limitations unless the answer is missing.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{formatted_contexts}\n\n"
        "Return only the answer."
    )


def _detect_language(question: str) -> str:
    q = question.lower()
    fr_markers = ["quoi", "pourquoi", "comment", "est-ce", "c'est", " le ", " la ", " les ", " des ", " du ", " un ", " une "]
    if any(marker in q for marker in fr_markers):
        return "fr"
    return "en"


def _no_context_message(language: str) -> str:
    if language == "fr":
        return "Je ne peux pas repondre a partir du contexte disponible."
    return "I cannot answer from the available context."


def _context_only_fallback(language: str, question: str, contexts: list[str]) -> str:
    snippet = _best_context_snippet(question, contexts)
    if not snippet:
        return _no_context_message(language)
    if language == "fr":
        return f"Reponse fondee sur le document: {snippet}"
    return f"Answer grounded in the document: {snippet}"


def _best_context_snippet(question: str, contexts: list[str], max_length: int = 320) -> str:
    if not contexts:
        return ""

    question_tokens = _tokenize(question)
    best_context = max(contexts, key=lambda context: _context_overlap(question_tokens, _tokenize(context)))
    snippet = best_context.strip()
    if len(snippet) <= max_length:
        return snippet
    return snippet[: max_length - 3].rstrip() + "..."


def _context_overlap(question_tokens: set[str], context_tokens: set[str]) -> int:
    if not question_tokens or not context_tokens:
        return 0
    return len(question_tokens.intersection(context_tokens))


def _looks_generic(text: str) -> bool:
    lowered = text.lower()
    generic_markers = [
        "based on the provided context",
        "from the provided context",
        "context only answer",
        "i cannot answer from the available context",
        "the document excerpts do not contain enough information",
        "as an ai",
        "here is a summary",
        "summary of the document",
    ]
    return any(marker in lowered for marker in generic_markers)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE) if token}
