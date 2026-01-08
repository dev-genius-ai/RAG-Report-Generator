"""
LangGraph nodes for the report generation workflow.
Each node represents a step in the RAG pipeline.
"""

from typing import List, Optional, Protocol

from langchain.schema import Document
from langchain_openai import ChatOpenAI

from ..config import Settings, get_settings
from ..observability.cost_tracker import CostTracker, get_cost_tracker
from ..observability.logger import get_logger
from ..retrieval.vector_store import VectorStore
from .state import AgentState

logger = get_logger(__name__)


class LLM(Protocol):
    """Protocol for chat models used by the agent nodes."""

    def invoke(self, prompt: str):  # pragma: no cover - protocol
        ...


def build_report_prompt(query: str, context: str, max_report_length: int) -> str:
    """Construct the prompt used to generate the detailed report."""
    return (
        "You are a senior business analyst tasked with creating a comprehensive report "
        "based on company data.\n\n"
        f"Query: {query}\n\n"
        "Context from company documents:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "1. Analyze the provided context thoroughly\n"
        "2. Generate a detailed, well-structured report that addresses the query\n"
        "3. Include specific data points and insights from the documents\n"
        "4. Organize the report with clear sections and headings\n"
        "5. Be objective and factual, citing information from the sources\n"
        "6. If the context doesn't fully answer the query, acknowledge the limitations\n\n"
        f"Generate a comprehensive report (aim for ~{max_report_length} tokens):"
    )


def build_summary_prompt(report: str) -> str:
    """Construct the prompt used to generate the executive summary."""
    return (
        "Create a concise executive summary (3-5 key bullet points) of the following report:\n\n"
        f"{report}\n\n"
        "Executive Summary:"
    )


class AgentNodes:
    """Collection of nodes for the report generation graph."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: Optional[LLM] = None,
        cost_tracker: Optional[CostTracker] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize agent nodes.

        Args:
            vector_store: VectorStore instance for retrieval
            llm: Optional chat model to use (defaults to ChatOpenAI)
            cost_tracker: Optional cost tracker (defaults to global tracker)
            settings: Optional settings instance (defaults to global settings)
        """
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.cost_tracker = cost_tracker or get_cost_tracker()

        # Initialize LLM
        self.llm: LLM = llm or ChatOpenAI(
            model=self.settings.openai_model,
            openai_api_key=self.settings.openai_api_key,
            temperature=0.7,
        )

    def retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents from the vector store.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents
        """
        logger.info("node_retrieve_start", query=state["query"])

        try:
            query = state["query"]
            
            # Retrieve documents with scores
            results = self.vector_store.similarity_search_with_score(
                query, k=self.settings.top_k_results
            )
            
            documents = [doc for doc, _ in results]
            scores = [float(score) for _, score in results]
            
            state["retrieved_documents"] = documents
            state["relevance_scores"] = scores
            
            logger.info(
                "node_retrieve_complete",
                num_documents=len(documents),
                avg_score=sum(scores) / len(scores) if scores else 0,
            )
            
            return state
            
        except Exception as e:
            logger.error("node_retrieve_failed", error=str(e))
            state["error"] = f"Document retrieval failed: {str(e)}"
            return state

    def build_context(self, state: AgentState) -> AgentState:
        """
        Build context from retrieved documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with context
        """
        logger.info("node_build_context_start")

        try:
            documents = state.get("retrieved_documents", [])
            
            if not documents:
                logger.warning("no_documents_for_context")
                state["context"] = ""
                state["sources"] = []
                return state
            
            # Combine document contents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", f"Document {i+1}")
                sources.append(source)
                
                context_parts.append(f"--- Source {i+1}: {source} ---\n{doc.page_content}\n")
            
            context = "\n".join(context_parts)
            
            state["context"] = context
            state["sources"] = sources
            
            logger.info(
                "node_build_context_complete",
                context_length=len(context),
                num_sources=len(sources),
            )
            
            return state
            
        except Exception as e:
            logger.error("node_build_context_failed", error=str(e))
            state["error"] = f"Context building failed: {str(e)}"
            return state

    def generate_report(self, state: AgentState) -> AgentState:
        """
        Generate detailed report from context.

        Args:
            state: Current agent state

        Returns:
            Updated state with generated report
        """
        logger.info("node_generate_report_start")

        try:
            query = state["query"]
            context = state.get("context", "")
            
            if not context:
                logger.warning("no_context_for_report")
                state["report"] = "No relevant information found to generate a report."
                return state
            
            # Create prompt
            prompt = build_report_prompt(query, context, self.settings.max_report_length)

            # Count input tokens
            input_tokens = self.cost_tracker.count_tokens(prompt, self.settings.openai_model)
            
            # Generate report
            response = self.llm.invoke(prompt)
            report = response.content
            
            # Count output tokens
            output_tokens = self.cost_tracker.count_tokens(report, self.settings.openai_model)
            
            # Track cost
            cost = self.cost_tracker.track_call(
                model=self.settings.openai_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate_report",
                metadata={"query": query},
            )
            
            state["report"] = report
            state["num_tokens_used"] = state.get("num_tokens_used", 0) + input_tokens + output_tokens
            state["total_cost"] = state.get("total_cost", 0.0) + cost
            
            logger.info(
                "node_generate_report_complete",
                report_length=len(report),
                tokens_used=input_tokens + output_tokens,
                cost=cost,
            )
            
            return state
            
        except Exception as e:
            logger.error("node_generate_report_failed", error=str(e))
            state["error"] = f"Report generation failed: {str(e)}"
            return state

    def generate_summary(self, state: AgentState) -> AgentState:
        """
        Generate executive summary from the report.

        Args:
            state: Current agent state

        Returns:
            Updated state with summary
        """
        logger.info("node_generate_summary_start")

        try:
            report = state.get("report", "")
            
            if not report:
                logger.warning("no_report_for_summary")
                state["summary"] = ""
                return state
            
            # Create prompt
            prompt = build_summary_prompt(report)

            # Count input tokens
            input_tokens = self.cost_tracker.count_tokens(prompt, self.settings.openai_model)
            
            # Generate summary
            response = self.llm.invoke(prompt)
            summary = response.content
            
            # Count output tokens
            output_tokens = self.cost_tracker.count_tokens(summary, self.settings.openai_model)
            
            # Track cost
            cost = self.cost_tracker.track_call(
                model=self.settings.openai_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate_summary",
            )
            
            state["summary"] = summary
            state["num_tokens_used"] = state.get("num_tokens_used", 0) + input_tokens + output_tokens
            state["total_cost"] = state.get("total_cost", 0.0) + cost
            
            logger.info(
                "node_generate_summary_complete",
                summary_length=len(summary),
                tokens_used=input_tokens + output_tokens,
                cost=cost,
            )
            
            return state
            
        except Exception as e:
            logger.error("node_generate_summary_failed", error=str(e))
            state["error"] = f"Summary generation failed: {str(e)}"
            return state

