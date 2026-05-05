import logging

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    if not settings.service_token:
        return {}
    return {"Authorization": f"Bearer {settings.service_token}"}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_and_query() -> None:
    ingest_response = client.post(
        "/ingest",
        headers=_auth_headers(),
        json={
            "userId": "u1",
            "documentId": "d1",
            "content": "SmartDoc stores and retrieves context for answers.",
        },
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunksCreated"] >= 1

    query_response = client.post(
        "/query",
        headers=_auth_headers(),
        json={"userId": "u1", "question": "What does SmartDoc do?", "documentIds": ["d1"]},
    )
    assert query_response.status_code == 200
    assert "answer" in query_response.json()


def test_query_falls_back_to_top_chunks_when_no_lexical_overlap() -> None:
    ingest_response = client.post(
        "/ingest",
        headers=_auth_headers(),
        json={
            "userId": "u2",
            "documentId": "d2",
            "content": "Spring Boot works with FastAPI in SmartDoc architecture.",
        },
    )
    assert ingest_response.status_code == 200

    query_response = client.post(
        "/query",
        headers=_auth_headers(),
        json={"userId": "u2", "question": "zzzxxyyq", "documentIds": ["d2"]},
    )
    assert query_response.status_code == 200

    body = query_response.json()
    assert body["answer"] != "I cannot answer from the available context."
    assert len(body["citations"]) > 0


def test_service_token_is_enforced_and_audited(monkeypatch, caplog) -> None:
    monkeypatch.setattr(settings, "service_token", "shared-token")

    with caplog.at_level(logging.INFO, logger="smartdoc_ai.audit"):
        unauthorized = client.post(
            "/ingest",
            json={
                "userId": "u-secure",
                "documentId": "d-secure",
                "content": "Token protected content for auditing.",
            },
        )
        assert unauthorized.status_code == 401

        authorized = client.post(
            "/ingest",
            headers={"Authorization": "Bearer shared-token"},
            json={
                "userId": "u-secure",
                "documentId": "d-secure",
                "content": "Token protected content for auditing.",
            },
        )
        assert authorized.status_code == 200

        query = client.post(
            "/query",
            headers={"Authorization": "Bearer shared-token"},
            json={"userId": "u-secure", "question": "What is protected?", "documentIds": ["d-secure"]},
        )
        assert query.status_code == 200

    assert "event=unauthorized_access" in caplog.text
    assert "event=ingest_received" in caplog.text
    assert "event=ingest_completed" in caplog.text
    assert "event=query_received" in caplog.text
    assert "event=query_completed" in caplog.text


