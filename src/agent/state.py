"""
Agent state definition for LangGraph.
Defines the data structure passed between nodes in the graph.
"""

from typing import Annotated, List, Optional

from langchain.schema import Document
from typing_extensions import TypedDict


def reduce_documents(existing: Optional[List[Document]], new: List[Document]) -> List[Document]:
    """Reducer function for accumulating documents."""
    if existing is None:
        return new
    return existing + new


class AgentState(TypedDict):
    """
    State for the report generation agent.
    
    This state is passed between nodes in the LangGraph workflow.
    """
    
    # Input
    query: str
    """The user's query or question about the company data"""
    
    # Retrieval results
    retrieved_documents: Annotated[List[Document], reduce_documents]
    """Documents retrieved from the vector store"""
    
    # Analysis
    relevance_scores: Optional[List[float]]
    """Relevance scores for retrieved documents"""
    
    context: Optional[str]
    """Combined context from relevant documents"""
    
    # Report generation
    report: Optional[str]
    """Generated report"""
    
    summary: Optional[str]
    """Executive summary of the report"""
    
    # Metadata
    sources: Optional[List[str]]
    """Source documents used in the report"""
    
    num_tokens_used: int
    """Total tokens used in generation"""
    
    total_cost: float
    """Total cost of API calls"""
    
    # Error handling
    error: Optional[str]
    """Error message if any step fails"""

