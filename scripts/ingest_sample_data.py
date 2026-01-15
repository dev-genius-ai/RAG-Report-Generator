#!/usr/bin/env python
"""
Script to ingest sample company data into the vector store.
Run this script to populate the database with example data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.data_ingestion.ingestion_pipeline import IngestionPipeline
from src.observability.logger import setup_logging, get_logger
from src.retrieval.vector_store import VectorStore


def main():
    """Ingest sample data."""
    # Setup
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()
    
    logger.info("sample_data_ingestion_started")
    print("🚀 Starting sample data ingestion...")
    
    # Initialize components
    vector_store = VectorStore()
    pipeline = IngestionPipeline(vector_store)
    
    # Get data directory
    data_dir = Path(__file__).parent.parent / "data" / "raw_data"
    
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"\n📁 Data directory: {data_dir}")
    
    # Check existing data
    stats = pipeline.get_collection_stats()
    print(f"\n📊 Current collection: {stats['name']}")
    print(f"   Documents before ingestion: {stats['count']}")
    
    if stats['count'] > 0:
        response = input("\n⚠ Collection already has data. Clear it first? (yes/no): ")
        if response.lower() == "yes":
            pipeline.clear_collection()
            print("✓ Collection cleared")
    
    # Ingest data
    print("\n📥 Ingesting sample data...")
    try:
        num_chunks = pipeline.ingest_directory(str(data_dir))
        print(f"\n✓ Successfully ingested {num_chunks} chunks")
        
        # Show updated stats
        stats = pipeline.get_collection_stats()
        print(f"\n📊 Final collection statistics:")
        print(f"   Name: {stats['name']}")
        print(f"   Total documents: {stats['count']}")
        
        print("\n✅ Sample data ingestion complete!")
        print("\nYou can now run queries like:")
        print('  python main.py --query "What are the company\'s revenue trends?"')
        print('  python main.py --query "Summarize customer satisfaction metrics"')
        print('  python main.py --query "What products does ACME offer?"')
        
    except Exception as e:
        logger.error("sample_data_ingestion_failed", error=str(e))
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

