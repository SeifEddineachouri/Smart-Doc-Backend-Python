from google import genai
from google.genai import types
import logging
import re
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

_TRANSIENT_MARKERS = ("503", "429", "unavailable", "high demand", "resource_exhausted", "rate limit")
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 0.6


def _build_generate_config():
    """Generation config tuned for fast, grounded answers.

    The key lever is thinking_budget=0: gemini-2.5-flash enables an internal
    "thinking" pass by default, which adds several seconds per answer and can
    push the call past the gateway read-timeout (surfacing as a 503). Extractive
    document QA does not need it, so we turn it off for low latency.
    """
    base_kwargs = {"temperature": 0.2, "max_output_tokens": 800}

    thinking_config_cls = getattr(types, "ThinkingConfig", None)
    if thinking_config_cls is not None:
        try:
            return types.GenerateContentConfig(
                thinking_config=thinking_config_cls(thinking_budget=0),
                **base_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - older google-genai without thinking support
            logger.warning("thinking_config unsupported by installed google-genai (%s); using default config", exc)

    try:
        return types.GenerateContentConfig(**base_kwargs)
    except Exception:  # noqa: BLE001 - fall back to library defaults if config shape changed
        return None


class GeminiClient:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None
        self._generate_config = _build_generate_config()

    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        language = _detect_language(question)

        if not contexts:
            return _no_context_message(language)

        if self._client is None:
            return _context_only_fallback(language, question, contexts)

        prompt = _build_prompt(question, contexts)

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                if self._generate_config is not None:
                    response = self._client.models.generate_content(
                        model=settings.model_name, contents=prompt, config=self._generate_config
                    )
                else:
                    response = self._client.models.generate_content(model=settings.model_name, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    # A model that correctly reports the answer is absent is a
                    # VALID response and passes through unchanged.
                    return text
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if _is_transient(exc) and attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_BACKOFF_SECONDS * (2**attempt))
                    continue
                break

        if last_error is not None:
            logger.warning("Gemini generate_content failed (model=%s): %s", settings.model_name, last_error)
            if _is_rate_limited(last_error):
                # Quota/rate-limit is a transient provider state, not a document
                # problem. Tell the user plainly instead of dumping raw context
                # that looks like a low-quality answer.
                return _rate_limited_message(language)
        # Offline or unexpected error: surface the most relevant excerpt so the
        # user still gets something grounded in their document.
        return _context_only_fallback(language, question, contexts)


def _build_prompt(question: str, contexts: list[str]) -> str:
    formatted_contexts = "\n".join(f"- {context}" for context in contexts)
    return (
        "You are a precise document question-answering assistant.\n"
        "Use only the provided document excerpts. Do not invent facts.\n"
        "Answer the exact question asked, not a generic summary.\n"
        "Be concrete and specific: state the clause, fact, value, date, condition, or obligation that directly answers the question.\n"
        "Reply in the SAME language as the question.\n"
        "If the excerpts do not contain the answer, say plainly that the document does not contain enough information to answer this question.\n"
        "Keep the answer focused: at most 5 sentences.\n"
        "Write clean prose only: do NOT include bracketed reference tags, document ids, or chunk indices (e.g. [document_id=...]) in your answer.\n"
        "Do not mention that you are an AI model or talk about limitations unless the answer is genuinely missing.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{formatted_contexts}\n\n"
        "Answer:"
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


def _is_transient(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _is_rate_limited(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("429", "resource_exhausted", "quota", "rate limit"))


def _rate_limited_message(language: str) -> str:
    if language == "fr":
        return (
            "Le service d'IA est temporairement surchargé (limite de requêtes atteinte). "
            "Veuillez réessayer dans une minute."
        )
    return (
        "The AI service is temporarily rate-limited (request quota reached). "
        "Please try again in a minute."
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE) if token}
