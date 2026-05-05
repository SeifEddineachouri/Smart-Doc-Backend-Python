import logging

from fastapi import FastAPI

from app.core.config import settings
from app.api.routes import router


logger = logging.getLogger(__name__)

app = FastAPI(title="smartdoc-ai", version="0.1.0")
app.include_router(router)


@app.on_event("startup")
def log_runtime_configuration() -> None:
    logger.info(
        "smartdoc-ai startup: model=%s, provider_key_configured=%s, service_token_configured=%s, env_file=%s",
        settings.model_name,
        bool(settings.gemini_api_key),
        bool(settings.service_token),
        settings.env_file_path,
    )


