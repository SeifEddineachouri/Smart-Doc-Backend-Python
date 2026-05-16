from types import SimpleNamespace

from app.models.schemas import QueryResponse
from app.services.gemini_client import GeminiClient
from app.services.query_service import QueryService
from app.services.vector_store import Chunk


class _FakeVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def retrieve(self, user_id: str, question: str, document_ids: list[str] | None, top_k: int) -> list[Chunk]:
        self.calls.append(
            {
                "user_id": user_id,
                "question": question,
                "document_ids": document_ids,
                "top_k": top_k,
            }
        )
        return self.chunks


class _FakeModels:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def generate_content(self, model: str, contents: str) -> SimpleNamespace:
        self.calls.append({"model": model, "contents": contents})
        return SimpleNamespace(text=self.response_text)


class _FakeGeminiApi:
    def __init__(self, response_text: str) -> None:
        self.models = _FakeModels(response_text)


def test_query_service_formats_context_with_document_metadata() -> None:
    chunk = Chunk(
        user_id="user-1",
        document_id="contract-v3",
        chunk_index=2,
        content="The supplier is liable for direct damages only.",
        embedding=[0.1, 0.2],
    )
    vector_store = _FakeVectorStore([chunk])
    gemini = GeminiClient()
    gemini._client = _FakeGeminiApi("The supplier is liable for direct damages only.")

    response: QueryResponse = QueryService(vector_store, gemini).answer(
        user_id="user-1",
        question="What liabilities are missing?",
        document_ids=["contract-v3"],
    )

    assert response.answer == "The supplier is liable for direct damages only."
    assert response.citations[0].documentId == "contract-v3"
    assert vector_store.calls[0]["document_ids"] == ["contract-v3"]

    prompt = gemini._client.models.calls[0]["contents"]
    assert "Answer the exact question, not a generic summary." in prompt
    assert "If the excerpts do not contain the answer" in prompt
    assert "[document_id=contract-v3 chunk_index=2]" in prompt
    assert "The supplier is liable for direct damages only." in prompt


def test_generic_model_answer_falls_back_to_document_excerpt() -> None:
    gemini = GeminiClient()
    gemini._client = _FakeGeminiApi("Based on the provided context, the answer is not explicit.")

    answer = gemini.answer_with_context(
        question="What liabilities are missing?",
        contexts=[
            "[document_id=contract-v3 chunk_index=2] The supplier is liable for direct damages only.",
            "[document_id=contract-v3 chunk_index=3] Indirect damages are excluded.",
        ],
    )

    assert answer.startswith("Answer grounded in the document:")
    assert "direct damages only" in answer or "Indirect damages are excluded" in answer
    assert "Based on the provided context" not in answer


def test_no_context_returns_clear_message() -> None:
    gemini = GeminiClient()

    assert gemini.answer_with_context(question="What is missing?", contexts=[]) == "I cannot answer from the available context."

