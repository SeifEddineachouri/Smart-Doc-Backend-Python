from app.core.config import settings
from app.models.schemas import Citation, QueryResponse
from app.services.gemini_client import GeminiClient
from app.services.vector_store import InMemoryVectorStore


class QueryService:
    def __init__(self, vector_store: InMemoryVectorStore, gemini_client: GeminiClient) -> None:
        self._vector_store = vector_store
        self._gemini_client = gemini_client

    def answer(self, user_id: str, question: str, document_ids: list[str] | None) -> QueryResponse:
        chunks = self._vector_store.retrieve(
            user_id=user_id,
            question=question,
            document_ids=document_ids,
            top_k=settings.retrieval_top_k,
        )

        contexts = [
            f"[document_id={chunk.document_id} chunk_index={chunk.chunk_index}] {chunk.content}"
            for chunk in chunks
        ]
        answer = self._gemini_client.answer_with_context(question=question, contexts=contexts)
        citations = [
            Citation(documentId=chunk.document_id, chunkIndex=chunk.chunk_index, snippet=chunk.content[:240])
            for chunk in chunks
        ]
        return QueryResponse(answer=answer, citations=citations)

