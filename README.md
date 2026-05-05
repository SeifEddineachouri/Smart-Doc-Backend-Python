# smartdoc-ai

FastAPI microservice MVP for SmartDoc RAG workflows.

## Service ownership (to avoid duplicate work)

- Owns: ingestion, retrieval, Gemini answer generation.
- Does not own: sign-in/sign-up/JWT, refresh cookies, frontend-facing auth APIs.
- Does not persist Spring chat history tables; it returns AI responses to the Spring gateway.

## Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`

## Environment variables

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` or `OPENAI_API_KEY` (first non-empty value is used)
- `SERVICE_TOKEN` (or `APP_AI_SERVICE_TOKEN` / `APP_AI_GATEWAY_SERVICE_TOKEN`)
- `MODEL_NAME` or `GEMINI_MODEL_NAME` (default: `gemini-3.1`)
- `CHUNK_SIZE_WORDS` (default: `220`)
- `CHUNK_OVERLAP_WORDS` (default: `30`)
- `RETRIEVAL_TOP_K` (default: `5`)
- `SMARTDOC_AI_ENV_FILE` (optional absolute path to a custom `.env` file)

Use `smartdoc-ai/.env.example` as template, then create your local `smartdoc-ai/.env`.

## Run locally

```powershell
Set-Location "C:\Users\seifa\Documents\smartdoc\smartdoc-ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

You can start from the repository root as well; config still resolves `smartdoc-ai/.env` automatically.

```powershell
Set-Location "C:\Users\seifa\Documents\smartdoc"
python -m uvicorn smartdoc-ai.app.main:app --reload --port 8000
```

## Quick test

```powershell
Set-Location "C:\Users\seifa\Documents\smartdoc\smartdoc-ai"
pytest -q
```

## Health check

`GET /health` returns runtime config hints:

- `status`
- `providerConfigured` (whether an API key is loaded)
- `modelName`

