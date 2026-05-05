from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    providerConfigured: bool = False
    modelName: str


class IngestRequest(BaseModel):
    userId: str = Field(min_length=1)
    documentId: str = Field(min_length=1)
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    status: str
    chunksCreated: int


class QueryRequest(BaseModel):
    userId: str = Field(min_length=1)
    question: str = Field(min_length=1)
    documentIds: list[str] | None = None


class Citation(BaseModel):
    documentId: str
    chunkIndex: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

