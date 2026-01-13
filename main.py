"""
Main entry point for the RAG Company Report Generator.
Provides CLI interface for generating reports from company data.
"""

import argparse
import sys
from pathlib import Path

from src.agent.graph import ReportGenerationGraph
from src.config import get_settings
from src.data_ingestion.ingestion_pipeline import IngestionPipeline
from src.observability.cost_tracker import get_cost_tracker
from src.observability.logger import get_logger, setup_logging
from src.report_writer import ReportWriter
from src.retrieval.vector_store import VectorStore


def setup_environment():
    """Setup logging and configuration."""
    setup_logging()
    settings = get_settings()
    return settings


def ingest_data(args, vector_store: VectorStore):
    """Handle data ingestion."""
    logger = get_logger(__name__)
    pipeline = IngestionPipeline(vector_store)
    
    if args.ingest_file:
        logger.info("ingesting_file", file_path=args.ingest_file)
        num_chunks = pipeline.ingest_file(args.ingest_file)
        print(f"✓ Successfully ingested {num_chunks} chunks from {args.ingest_file}")
    
    elif args.ingest_dir:
        logger.info("ingesting_directory", directory_path=args.ingest_dir)
        num_chunks = pipeline.ingest_directory(args.ingest_dir)
        print(f"✓ Successfully ingested {num_chunks} chunks from {args.ingest_dir}")
    
    # Show collection stats
    stats = pipeline.get_collection_stats()
    print(f"\nCollection Statistics:")
    print(f"  Name: {stats['name']}")
    print(f"  Total Documents: {stats['count']}")


def generate_report(args, vector_store: VectorStore):
    """Handle report generation."""
    logger = get_logger(__name__)
    cost_tracker = get_cost_tracker()
    
    # Check if vector store has data
    stats = vector_store.get_collection_info()
    if stats['count'] == 0:
        print("⚠ Warning: Vector store is empty. Please ingest data first.")
        print("  Use: python main.py --ingest-dir <directory>")
        return
    
    print(f"📊 Vector store contains {stats['count']} document chunks")
    
    # Initialize graph
    graph = ReportGenerationGraph(vector_store)
    
    # Generate report
    query = args.query
    print(f"\n🔍 Query: {query}")
    print("\n⏳ Generating report...\n")
    
    result = graph.generate_report(query)
    
    # Check for errors
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)
    
    # Display results
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(result.get("summary", "No summary generated"))
    
    print("\n" + "=" * 80)
    print("DETAILED REPORT")
    print("=" * 80)
    print(result.get("report", "No report generated"))
    
    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)
    sources = result.get("sources", [])
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source}")
    
    # Save to file if requested
    if args.output:
        writer = ReportWriter()
        
        metadata = {
            "tokens_used": result.get("num_tokens_used", 0),
            "cost_usd": f"${result.get('total_cost', 0.0):.4f}",
            "num_sources": len(sources),
        }
        
        if args.format == "markdown":
            file_path = writer.save_report_markdown(
                query=query,
                report=result.get("report", ""),
                summary=result.get("summary", ""),
                sources=sources,
                metadata=metadata,
            )
        else:
            file_path = writer.save_report(
                query=query,
                report=result.get("report", ""),
                summary=result.get("summary", ""),
                sources=sources,
                metadata=metadata,
            )
        
        print(f"\n📝 Report saved to: {file_path}")
    
    # Show cost summary
    print("\n" + "=" * 80)
    print("COST SUMMARY")
    print("=" * 80)
    summary = cost_tracker.get_session_summary()
    print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
    print(f"Input Tokens: {summary['total_input_tokens']:,}")
    print(f"Output Tokens: {summary['total_output_tokens']:,}")
    print("\nBy Model:")
    for model, data in summary['by_model'].items():
        print(f"  {model}: ${data['cost_usd']:.4f}")


def show_stats(vector_store: VectorStore):
    """Show collection statistics."""
    stats = vector_store.get_collection_info()
    print("\n📊 Collection Statistics:")
    print(f"  Name: {stats['name']}")
    print(f"  Total Documents: {stats['count']}")
    if stats['metadata']:
        print(f"  Metadata: {stats['metadata']}")


def clear_data(vector_store: VectorStore):
    """Clear all data from the vector store."""
    logger = get_logger(__name__)
    
    response = input("⚠ Are you sure you want to clear all data? (yes/no): ")
    if response.lower() == "yes":
        pipeline = IngestionPipeline(vector_store)
        pipeline.clear_collection()
        print("✓ Collection cleared successfully")
        logger.info("collection_cleared_by_user")
    else:
        print("Operation cancelled")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Company Report Generator - Generate reports from company data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest data from a directory
  python main.py --ingest-dir ./data/raw_data
  
  # Ingest a single file
  python main.py --ingest-file ./data/company_report.pdf
  
  # Generate a report
  python main.py --query "What are the company's revenue trends?"
  
  # Generate and save report
  python main.py --query "Summarize Q4 performance" --output --format markdown
  
  # Show collection stats
  python main.py --stats
  
  # Clear all data
  python main.py --clear
        """
    )
    
    # Ingestion options
    ingest_group = parser.add_mutually_exclusive_group()
    ingest_group.add_argument(
        "--ingest-file",
        type=str,
        help="Ingest a single file into the vector store"
    )
    ingest_group.add_argument(
        "--ingest-dir",
        type=str,
        help="Ingest all supported files from a directory"
    )
    
    # Query options
    parser.add_argument(
        "--query",
        type=str,
        help="Query to generate a report for"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        action="store_true",
        help="Save the report to a file"
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format for saved report (default: text)"
    )
    
    # Utility options
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all data from the vector store"
    )
    
    args = parser.parse_args()
    
    # Setup
    settings = setup_environment()
    logger = get_logger(__name__)
    
    logger.info("application_started", args=vars(args))
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Handle commands
    try:
        if args.ingest_file or args.ingest_dir:
            ingest_data(args, vector_store)
        
        elif args.query:
            generate_report(args, vector_store)
        
        elif args.stats:
            show_stats(vector_store)
        
        elif args.clear:
            clear_data(vector_store)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error("application_error", error=str(e))
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    
    finally:
        # Log cost summary
        cost_tracker = get_cost_tracker()
        cost_tracker.log_session_summary()


if __name__ == "__main__":
    main()

