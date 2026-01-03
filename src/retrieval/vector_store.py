"""
Vector store management using ChromaDB.
Handles document ingestion, retrieval, and persistence.
"""

from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from ..config import get_settings
from ..observability.logger import get_logger
from .embeddings import get_embeddings

logger = get_logger(__name__)


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimal protocol for Chroma-like vector stores used by the agent."""

    def add_documents(self, documents: List[Document]) -> List[str]:  # pragma: no cover - protocol
        ...

    def similarity_search(self, query: str, k: int) -> List[Document]:  # pragma: no cover - protocol
        ...

    def similarity_search_with_score(  # pragma: no cover - protocol
        self, query: str, k: int
    ) -> List[tuple[Document, float]]:
        ...

    def as_retriever(self, *args, **kwargs):  # pragma: no cover - protocol
        ...


class VectorStore:
    """
    Vector store wrapper for ChromaDB.
    Provides high-level interface for document storage and retrieval.
    """

    def __init__(
        self,
        client: Optional[chromadb.PersistentClient] = None,
        vectorstore: Optional[VectorStoreBackend] = None,
    ):
        """
        Initialize the vector store.

        The client and underlying vector store can be injected for tests to avoid
        touching the real Chroma persistence layer.
        """
        self.settings = get_settings()
        self.embeddings = get_embeddings()

        # Ensure persistence directory exists when we manage the client ourselves.
        persist_dir = Path(self.settings.chroma_persist_directory)
        if client is None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))

        self.client: chromadb.PersistentClient = client

        # Initialize or use provided vector store implementation
        self.vectorstore: VectorStoreBackend
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        """Initialize or load existing vector store."""
        try:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.settings.chroma_collection_name,
                embedding_function=self.embeddings,
            )

            # Check if collection exists and has documents
            collection = self.client.get_collection(self.settings.chroma_collection_name)
            doc_count = collection.count()

            logger.info(
                "vectorstore_initialized",
                collection_name=self.settings.chroma_collection_name,
                document_count=doc_count,
            )
        except Exception as e:
            logger.warning("vectorstore_init_warning", error=str(e))
            # Create new collection
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.settings.chroma_collection_name,
                embedding_function=self.embeddings,
            )
            logger.info("vectorstore_created", collection_name=self.settings.chroma_collection_name)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("no_documents_to_add")
            return []

        try:
            ids = self.vectorstore.add_documents(documents)
            logger.info("documents_added", count=len(documents), ids_count=len(ids))
            return ids
        except Exception as e:
            logger.error("documents_add_failed", error=str(e), count=len(documents))
            raise

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return (defaults to settings.top_k_results)

        Returns:
            List of similar documents
        """
        if k is None:
            k = self.settings.top_k_results

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(
                "similarity_search_complete",
                query_length=len(query),
                k=k,
                results_found=len(results),
            )
            return results
        except Exception as e:
            logger.error("similarity_search_failed", error=str(e), query_length=len(query))
            raise

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if k is None:
            k = self.settings.top_k_results

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(
                "similarity_search_with_score_complete",
                query_length=len(query),
                k=k,
                results_found=len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "similarity_search_with_score_failed",
                error=str(e),
                query_length=len(query),
            )
            raise

    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever interface for the vector store.

        Args:
            k: Number of results to return

        Returns:
            VectorStoreRetriever instance
        """
        if k is None:
            k = self.settings.top_k_results

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.settings.chroma_collection_name)
            logger.info("collection_deleted", collection_name=self.settings.chroma_collection_name)
            # Reinitialize
            self._initialize_vectorstore()
        except Exception as e:
            logger.error("collection_delete_failed", error=str(e))
            raise

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection metadata
        """
        try:
            collection = self.client.get_collection(self.settings.chroma_collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata or {},
            }
        except Exception as e:
            logger.error("get_collection_info_failed", error=str(e))
            return {"name": self.settings.chroma_collection_name, "count": 0, "metadata": {}}

