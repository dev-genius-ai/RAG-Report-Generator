"""Tests for vector store functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document

from src.retrieval.vector_store import VectorStore, VectorStoreBackend


@pytest.fixture
def settings():
    """Mock settings for testing."""
    with patch("src.retrieval.vector_store.get_settings") as mock:
        settings = Mock()
        settings.chroma_persist_directory = "./test_data/chroma_db"
        settings.chroma_collection_name = "test_collection"
        settings.top_k_results = 3
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_backend():
    """Fake Chroma-like backend for unit tests."""
    backend = Mock(spec=VectorStoreBackend)
    backend.add_documents.return_value = ["id1", "id2"]
    backend.similarity_search.return_value = [
        Document(page_content="result 1", metadata={}),
        Document(page_content="result 2", metadata={}),
    ]
    backend.similarity_search_with_score.return_value = [
        (Document(page_content="result 1", metadata={}), 0.1),
        (Document(page_content="result 2", metadata={}), 0.2),
    ]
    backend.as_retriever.return_value = Mock()
    return backend


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(page_content="This is a test document about AI.", metadata={"source": "test1.txt"}),
        Document(page_content="Machine learning is a subset of AI.", metadata={"source": "test2.txt"}),
        Document(page_content="Deep learning uses neural networks.", metadata={"source": "test3.txt"}),
    ]


def test_vector_store_initialization_uses_backend_injection(settings, mock_backend):
    """VectorStore should use an injected backend without touching Chroma."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    # Client should be constructed once, and provided backend should be used.
    client_cls.assert_called_once()
    assert store.vectorstore is mock_backend


def test_add_documents_with_empty_list_is_noop(settings, mock_backend, caplog):
    """Adding no documents should be a no-op and return an empty list."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    with caplog.at_level("WARNING"):
        ids = store.add_documents([])

    assert ids == []
    assert "no_documents_to_add" in caplog.text
    mock_backend.add_documents.assert_not_called()


def test_add_documents_delegates_to_backend(settings, mock_backend, sample_documents):
    """Adding documents delegates to the backend and logs success."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    ids = store.add_documents(sample_documents)

    mock_backend.add_documents.assert_called_once_with(sample_documents)
    assert ids == ["id1", "id2"]


def test_similarity_search_uses_default_k(settings, mock_backend):
    """Similarity search uses settings.top_k_results when k is not provided."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    results = store.similarity_search("query text")

    mock_backend.similarity_search.assert_called_once_with("query text", k=settings.top_k_results)
    assert len(results) == 2


def test_similarity_search_with_score_uses_default_k(settings, mock_backend):
    """Similarity search with score uses settings.top_k_results when k is not provided."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    results = store.similarity_search_with_score("query text")

    mock_backend.similarity_search_with_score.assert_called_once_with(
        "query text", k=settings.top_k_results
    )
    assert len(results) == 2
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


def test_get_retriever_uses_backend(settings, mock_backend):
    """get_retriever should delegate to the backend retriever factory."""
    with patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls, patch(
        "src.retrieval.vector_store.get_embeddings"
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    retriever = store.get_retriever()

    mock_backend.as_retriever.assert_called_once()
    assert retriever is mock_backend.as_retriever.return_value

