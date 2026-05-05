import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import settings
from app.models.schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.services.embedding_client import EmbeddingClient
from app.services.gemini_client import GeminiClient
from app.services.ingest_service import chunk_text
from app.services.query_service import QueryService
from app.services.vector_store import InMemoryVectorStore


router = APIRouter()
audit_logger = logging.getLogger("smartdoc_ai.audit")
_embedding_client = EmbeddingClient()
_vector_store = InMemoryVectorStore(_embedding_client)
_query_service = QueryService(_vector_store, GeminiClient())


def _check_service_token(request: Request) -> None:
    if not settings.service_token:
        return

    auth_header = request.headers.get("authorization", "")
    if auth_header != f"Bearer {settings.service_token}":
        audit_logger.warning("event=unauthorized_access path=%s method=%s", request.url.path, request.method)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", providerConfigured=bool(settings.gemini_api_key), modelName=settings.model_name)


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(_check_service_token)])
def ingest(payload: IngestRequest) -> IngestResponse:
    audit_logger.info(
        "event=ingest_received userId=%s documentId=%s contentChars=%s",
        payload.userId,
        payload.documentId,
        len(payload.content),
    )
    chunks = chunk_text(payload.content)
    chunks_created = _vector_store.upsert_document(
        user_id=payload.userId,
        document_id=payload.documentId,
        chunk_texts=chunks,
    )
    audit_logger.info(
        "event=ingest_completed userId=%s documentId=%s chunksCreated=%s",
        payload.userId,
        payload.documentId,
        chunks_created,
    )
    return IngestResponse(status="ingested", chunksCreated=chunks_created)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(_check_service_token)])
def query(payload: QueryRequest) -> QueryResponse:
    audit_logger.info(
        "event=query_received userId=%s questionChars=%s documentCount=%s",
        payload.userId,
        len(payload.question),
        0 if payload.documentIds is None else len(payload.documentIds),
    )
    response = _query_service.answer(
        user_id=payload.userId,
        question=payload.question,
        document_ids=payload.documentIds,
    )
    audit_logger.info(
        "event=query_completed userId=%s answerChars=%s citations=%s",
        payload.userId,
        len(response.answer),
        len(response.citations),
    )
    return response

