"""
Embeddings configuration and management.
Uses OpenAI embeddings with cost tracking.
"""

from typing import List

from langchain_openai import OpenAIEmbeddings

from ..config import get_settings
from ..observability.cost_tracker import get_cost_tracker
from ..observability.logger import get_logger

logger = get_logger(__name__)


class TrackedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings with cost tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cost_tracker = get_cost_tracker()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with cost tracking.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        # Count tokens
        total_tokens = sum(self._cost_tracker.count_tokens(text) for text in texts)
        
        # Get embeddings
        embeddings = super().embed_documents(texts)
        
        # Track cost
        self._cost_tracker.track_call(
            model=self.model,
            input_tokens=total_tokens,
            output_tokens=0,
            operation="embed_documents",
            metadata={"num_texts": len(texts)},
        )
        
        logger.debug(
            "documents_embedded",
            num_texts=len(texts),
            total_tokens=total_tokens,
        )
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed query with cost tracking.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        # Count tokens
        tokens = self._cost_tracker.count_tokens(text)
        
        # Get embedding
        embedding = super().embed_query(text)
        
        # Track cost
        self._cost_tracker.track_call(
            model=self.model,
            input_tokens=tokens,
            output_tokens=0,
            operation="embed_query",
            metadata={"query_length": len(text)},
        )
        
        logger.debug("query_embedded", tokens=tokens)
        
        return embedding


def get_embeddings() -> TrackedOpenAIEmbeddings:
    """
    Get configured embeddings instance.

    Returns:
        TrackedOpenAIEmbeddings instance
    """
    settings = get_settings()
    
    return TrackedOpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )

