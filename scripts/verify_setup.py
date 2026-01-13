#!/usr/bin/env python
"""
Setup verification script.
Checks that all components are properly configured and working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check environment configuration."""
    print("🔍 Checking environment configuration...")
    
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Check for API key
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            print("  ❌ OPENAI_API_KEY not configured in .env")
            return False
        
        print("  ✓ Environment configured")
        print(f"    - Model: {settings.openai_model}")
        print(f"    - Embedding Model: {settings.openai_embedding_model}")
        print(f"    - Environment: {settings.environment}")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "langchain",
        "langchain_openai",
        "langgraph",
        "chromadb",
        "structlog",
        "pydantic",
        "tiktoken",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print("     Run: pip install -r requirements.txt")
        return False
    
    print("  ✓ All dependencies installed")
    return True


def check_directories():
    """Check required directories exist."""
    print("\n🔍 Checking directory structure...")
    
    required_dirs = [
        "src",
        "src/agent",
        "src/config",
        "src/data_ingestion",
        "src/observability",
        "src/retrieval",
        "src/utils",
        "data",
        "data/raw_data",
        "logs",
        "reports",
        "scripts",
    ]
    
    base_dir = Path(__file__).parent.parent
    missing = []
    
    for dir_path in required_dirs:
        if not (base_dir / dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        print(f"  ❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("  ✓ Directory structure correct")
    return True


def check_sample_data():
    """Check if sample data exists."""
    print("\n🔍 Checking sample data...")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw_data"
    
    if not data_dir.exists():
        print("  ❌ Sample data directory not found")
        return False
    
    sample_files = list(data_dir.glob("*.txt"))
    
    if not sample_files:
        print("  ⚠ No sample data files found")
        print("    Sample data exists but you can add your own documents")
        return True
    
    print(f"  ✓ Sample data found ({len(sample_files)} files)")
    for file in sample_files:
        print(f"    - {file.name}")
    return True


def check_vector_store():
    """Check vector store initialization."""
    print("\n🔍 Checking vector store...")
    
    try:
        from src.retrieval.vector_store import VectorStore
        vector_store = VectorStore()
        stats = vector_store.get_collection_info()
        
        print(f"  ✓ Vector store initialized")
        print(f"    - Collection: {stats['name']}")
        print(f"    - Documents: {stats['count']}")
        
        if stats['count'] == 0:
            print("    ℹ  Vector store is empty - run ingestion to add documents")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store error: {e}")
        return False


def check_logging():
    """Check logging configuration."""
    print("\n🔍 Checking logging configuration...")
    
    try:
        from src.observability.logger import setup_logging, get_logger
        setup_logging()
        logger = get_logger("verify_setup")
        logger.info("test_log", test=True)
        
        print("  ✓ Logging configured")
        return True
        
    except Exception as e:
        print(f"  ❌ Logging error: {e}")
        return False


def check_cost_tracking():
    """Check cost tracking."""
    print("\n🔍 Checking cost tracking...")
    
    try:
        from src.observability.cost_tracker import get_cost_tracker
        tracker = get_cost_tracker()
        
        # Test token counting
        tokens = tracker.count_tokens("This is a test sentence.")
        
        if tokens > 0:
            print("  ✓ Cost tracking operational")
            return True
        else:
            print("  ❌ Token counting failed")
            return False
        
    except Exception as e:
        print(f"  ❌ Cost tracking error: {e}")
        return False


def run_all_checks():
    """Run all verification checks."""
    print("=" * 80)
    print("RAG Company Report Generator - Setup Verification")
    print("=" * 80)
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Sample Data", check_sample_data),
        ("Vector Store", check_vector_store),
        ("Logging", check_logging),
        ("Cost Tracking", check_cost_tracking),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for (name, _), result in zip(checks, results):
        status = "✓" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Ingest data: python scripts/ingest_sample_data.py")
        print("  2. Generate report: python main.py --query 'Your question'")
        print("  3. See QUICKSTART.md for more examples")
        return True
    else:
        print("\n⚠  Some checks failed. Please resolve the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)

