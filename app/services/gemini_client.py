from google import genai

from app.core.config import settings


class GeminiClient:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        language = _detect_language(question)

        if not contexts:
            return _no_context_message(language)

        if self._client is None:
            return _context_only_fallback(language, contexts[0][:600])

        prompt = (
            "Answer using only the provided context. "
            "If the answer is missing, say you cannot answer from context.\n\n"
            "Respond in the same language as the question.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n---\n".join(contexts)
        )

        try:
            response = self._client.models.generate_content(model=settings.model_name, contents=prompt)
            text = (response.text or "").strip()
            return text or _context_only_fallback(language, contexts[0][:600])
        except Exception:
            # Keep API responsive even when provider returns transient errors.
            return _context_only_fallback(language, contexts[0][:600])


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


def _context_only_fallback(language: str, snippet: str) -> str:
    if language == "fr":
        return f"Reponse basee sur le contexte (secours): {snippet}"
    return f"Context-only answer (fallback): {snippet}"
