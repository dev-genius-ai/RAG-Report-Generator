"""Tests for the FastAPI application."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import main as api_main


def _setup_app_with_fakes():
    """Configure the FastAPI app to use fake dependencies for testing."""
    fake_vector_store = Mock()
    fake_vector_store.get_collection_info.return_value = {
        "name": "test_collection",
        "count": 1,
        "metadata": {},
    }

    fake_ingestion_pipeline = Mock()
    fake_ingestion_pipeline.ingest_documents.return_value = 1
    fake_ingestion_pipeline.ingest_file.return_value = 1
    fake_ingestion_pipeline.clear_collection.return_value = None

    fake_report_graph = Mock()
    fake_report_graph.generate_report.return_value = {
        "query": "test",
        "report": "report body",
        "summary": "summary body",
        "sources": ["src1"],
        "num_tokens_used": 42,
        "total_cost": 0.1,
        "error": None,
    }

    api_main.create_vector_store = lambda: fake_vector_store
    api_main.create_ingestion_pipeline = lambda store: fake_ingestion_pipeline
    api_main.create_report_graph = lambda store: fake_report_graph

    return fake_vector_store, fake_ingestion_pipeline, fake_report_graph


def test_root_and_health_endpoints():
    _setup_app_with_fakes()

    with TestClient(api_main.app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        data = root_resp.json()
        assert data["message"].startswith("RAG Company Report Generator API")

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health = health_resp.json()
        assert health["status"] == "healthy"


def test_ingest_text_and_query_flow():
    fake_store, fake_pipeline, fake_graph = _setup_app_with_fakes()

    with TestClient(api_main.app) as client:
        ingest_resp = client.post(
            "/ingest/text",
            json={"text": "hello world", "source_name": "unit_test"},
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["chunks_added"] == 1

        fake_store.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 1,
            "metadata": {},
        }

        query_resp = client.post(
            "/query", json={"query": "test", "save_report": False, "format": "markdown"}
        )
        assert query_resp.status_code == 200
        body = query_resp.json()
        assert body["report"] == "report body"
        assert body["summary"] == "summary body"


def test_clear_collection_and_costs(monkeypatch):
    fake_store, fake_pipeline, _ = _setup_app_with_fakes()

    fake_cost_tracker = Mock()
    fake_cost_tracker.get_session_summary.return_value = {
        "total_cost_usd": 0.1,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "by_model": {"gpt-4-turbo-preview": {"cost_usd": 0.1, "tokens": {"input": 10, "output": 5}}},
    }
    monkeypatch.setattr("src.api.main.get_cost_tracker", lambda: fake_cost_tracker)

    with TestClient(api_main.app) as client:
        clear_resp = client.delete("/clear")
        assert clear_resp.status_code == 200
        fake_pipeline.clear_collection.assert_called_once()

        costs_resp = client.get("/costs")
        assert costs_resp.status_code == 200
        costs = costs_resp.json()
        assert costs["total_cost_usd"] == 0.1

