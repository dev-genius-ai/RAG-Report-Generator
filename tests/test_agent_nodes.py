"""Tests for AgentNodes behavior."""

from unittest.mock import Mock

import pytest
from langchain.schema import Document

from src.agent.nodes import (
    AgentNodes,
    build_report_prompt,
    build_summary_prompt,
)


class DummyResponse:
    """Simple container mimicking ChatOpenAI response objects."""

    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
def fake_settings():
    class SettingsStub:
        openai_model = "gpt-4-turbo-preview"
        openai_api_key = "test-key"
        max_report_length = 2000

    return SettingsStub()


@pytest.fixture
def fake_cost_tracker():
    tracker = Mock()
    tracker.count_tokens.side_effect = lambda text, model=None: len(text.split())
    tracker.track_call.return_value = 0.01
    return tracker


@pytest.fixture
def fake_vector_store():
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (Document(page_content="content 1", metadata={"source": "doc1"}), 0.1),
        (Document(page_content="content 2", metadata={"source": "doc2"}), 0.2),
    ]
    return store


@pytest.fixture
def fake_llm():
    llm = Mock()
    llm.invoke.side_effect = lambda prompt: DummyResponse(f"RESPONSE FOR: {prompt[:20]}")
    return llm


@pytest.fixture
def agent_nodes(fake_vector_store, fake_llm, fake_cost_tracker, fake_settings):
    return AgentNodes(
        vector_store=fake_vector_store,
        llm=fake_llm,
        cost_tracker=fake_cost_tracker,
        settings=fake_settings,
    )


def test_build_report_prompt_includes_query_and_context(fake_settings):
    prompt = build_report_prompt(
        query="What are revenues?",
        context="Revenue grew by 10%.",
        max_report_length=fake_settings.max_report_length,
    )
    assert "What are revenues?" in prompt
    assert "Revenue grew by 10%." in prompt
    assert str(fake_settings.max_report_length) in prompt


def test_build_summary_prompt_includes_report():
    report = "This is a long report."
    prompt = build_summary_prompt(report)
    assert report in prompt
    assert "Executive Summary" in prompt


def test_retrieve_documents_populates_state(agent_nodes, fake_vector_store):
    state = {"query": "test query"}

    updated = agent_nodes.retrieve_documents(state)

    fake_vector_store.similarity_search_with_score.assert_called_once()
    assert len(updated["retrieved_documents"]) == 2
    assert updated["relevance_scores"] == [0.1, 0.2]


def test_build_context_uses_sources_and_content(agent_nodes):
    docs = [
        Document(page_content="alpha", metadata={"source": "first.txt"}),
        Document(page_content="beta", metadata={"source": "second.txt"}),
    ]
    state = {"retrieved_documents": docs}

    updated = agent_nodes.build_context(state)

    assert "first.txt" in updated["context"]
    assert "second.txt" in updated["context"]
    assert updated["sources"] == ["first.txt", "second.txt"]


def test_build_context_handles_no_documents(agent_nodes):
    state = {"retrieved_documents": []}

    updated = agent_nodes.build_context(state)

    assert updated["context"] == ""
    assert updated["sources"] == []


def test_generate_report_updates_state_and_cost(agent_nodes, fake_cost_tracker, fake_llm):
    state = {
        "query": "test query",
        "context": "some context",
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    updated = agent_nodes.generate_report(state)

    fake_llm.invoke.assert_called_once()
    fake_cost_tracker.count_tokens.assert_any_call(
        updated["report"], agent_nodes.settings.openai_model
    )
    assert updated["report"].startswith("RESPONSE FOR:")
    assert updated["num_tokens_used"] > 0
    assert updated["total_cost"] > 0.0


def test_generate_report_handles_missing_context(agent_nodes, fake_llm):
    state = {"query": "test query", "context": ""}

    updated = agent_nodes.generate_report(state)

    # When context is empty, a default message should be set and LLM not invoked.
    fake_llm.invoke.assert_not_called()
    assert "No relevant information found" in updated["report"]


def test_generate_summary_updates_state(agent_nodes, fake_llm, fake_cost_tracker):
    state = {
        "report": "detailed report content",
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    updated = agent_nodes.generate_summary(state)

    fake_llm.invoke.assert_called_once()
    fake_cost_tracker.count_tokens.assert_any_call(
        updated["summary"], agent_nodes.settings.openai_model
    )
    assert updated["summary"].startswith("RESPONSE FOR:")
    assert updated["num_tokens_used"] > 0
    assert updated["total_cost"] > 0.0


def test_generate_summary_handles_missing_report(agent_nodes, fake_llm):
    state = {"report": ""}

    updated = agent_nodes.generate_summary(state)

    fake_llm.invoke.assert_not_called()
    assert updated["summary"] == ""

