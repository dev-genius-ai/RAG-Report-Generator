"""Tests for the ingestion pipeline."""

from unittest.mock import Mock

import pytest
from langchain.schema import Document

from src.data_ingestion.ingestion_pipeline import IngestionPipeline


@pytest.fixture
def fake_vector_store():
    """Fake VectorStore that records added documents."""
    store = Mock()
    store.add_documents.return_value = ["id1", "id2", "id3"]
    store.get_collection_info.return_value = {
        "name": "test_collection",
        "count": 3,
        "metadata": {},
    }
    store.delete_collection.return_value = None
    return store


@pytest.fixture
def fake_text_splitter():
    """Fake text splitter that returns pre-defined chunks."""
    splitter = Mock()

    def _split_documents(documents):
        chunks = []
        for idx, doc in enumerate(documents):
            chunks.append(
                Document(
                    page_content=f"chunk-{idx}-{doc.page_content}",
                    metadata=dict(doc.metadata),
                )
            )
        return chunks

    splitter.split_documents.side_effect = _split_documents
    return splitter


@pytest.fixture
def fake_document_loader():
    """Fake document loader for file and directory ingestion."""
    loader = Mock()
    loader.load_file.return_value = [
        Document(page_content="file content", metadata={"source": "file.txt"})
    ]
    loader.load_directory.return_value = [
        Document(page_content="dir content", metadata={"source": "dir.txt"})
    ]
    return loader


@pytest.fixture
def pipeline(fake_vector_store, fake_text_splitter, fake_document_loader):
    """Ingestion pipeline wired with fakes for unit testing."""
    return IngestionPipeline(
        vector_store=fake_vector_store,
        text_splitter=fake_text_splitter,
        document_loader=fake_document_loader,
    )


def test_ingest_file_adds_metadata_and_calls_vector_store(pipeline, fake_vector_store):
    """ingest_file should enrich chunks with metadata and delegate to vector store."""
    chunks_added = pipeline.ingest_file("some/path/report.pdf")

    # Fake splitter produces one chunk for the single loaded document.
    assert chunks_added == 1
    fake_vector_store.add_documents.assert_called_once()
    (chunks,) = fake_vector_store.add_documents.call_args.args
    assert len(chunks) == 1
    assert chunks[0].metadata["source_file"] == "some/path/report.pdf"
    assert chunks[0].metadata["ingestion_type"] == "file"


def test_ingest_directory_handles_no_documents_gracefully(fake_vector_store, fake_text_splitter):
    """ingest_directory should return 0 when no documents are found."""
    empty_loader = Mock()
    empty_loader.load_directory.return_value = []
    pipeline = IngestionPipeline(
        vector_store=fake_vector_store,
        text_splitter=fake_text_splitter,
        document_loader=empty_loader,
    )

    chunks_added = pipeline.ingest_directory("empty/dir")

    assert chunks_added == 0
    fake_vector_store.add_documents.assert_not_called()


def test_ingest_directory_adds_directory_metadata(pipeline, fake_vector_store):
    """ingest_directory should annotate chunks with directory metadata."""
    chunks_added = pipeline.ingest_directory("data/dir")

    assert chunks_added == 1
    fake_vector_store.add_documents.assert_called_once()
    (chunks,) = fake_vector_store.add_documents.call_args.args
    assert chunks[0].metadata["source_directory"] == "data/dir"
    assert chunks[0].metadata["ingestion_type"] == "directory"


def test_ingest_documents_marks_ingestion_type_direct(pipeline, fake_vector_store):
    """ingest_documents should mark chunks as direct ingestion."""
    documents = [
        Document(page_content="preloaded content", metadata={"source": "api"}),
    ]

    chunks_added = pipeline.ingest_documents(documents)

    assert chunks_added == 1
    fake_vector_store.add_documents.assert_called_once()
    (chunks,) = fake_vector_store.add_documents.call_args.args
    assert chunks[0].metadata["ingestion_type"] == "direct"


def test_get_collection_stats_delegates_to_vector_store(pipeline, fake_vector_store):
    """get_collection_stats should return the underlying vector store stats."""
    stats = pipeline.get_collection_stats()

    assert stats["name"] == "test_collection"
    assert stats["count"] == 3


def test_clear_collection_calls_vector_store_delete(pipeline, fake_vector_store):
    """clear_collection should delegate deletion to the vector store."""
    pipeline.clear_collection()

    fake_vector_store.delete_collection.assert_called_once()

