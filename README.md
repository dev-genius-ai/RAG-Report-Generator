# RAG Company Report Generator

A production-grade RAG (Retrieval-Augmented Generation) pipeline built with LangChain/LangGraph for retrieving company data from a vector database and generating comprehensive reports.

## 🏗️ Architecture

This project follows a clean, modular architecture with clear separation of concerns:

```
RAG/
├── src/
│   ├── agent/           # LangGraph workflow and nodes
│   ├── config/          # Configuration management
│   ├── data_ingestion/  # Document ingestion pipeline
│   ├── observability/   # Logging and cost tracking
│   ├── retrieval/       # Vector store and embeddings
│   └── utils/           # Document loaders and utilities
├── data/
│   ├── raw_data/        # Source documents
│   └── chroma_db/       # ChromaDB persistence
├── logs/                # Application logs
├── reports/             # Generated reports
├── scripts/             # Utility scripts
└── tests/               # Test suite
```

## ✨ Features

- **LangGraph State Machine**: Orchestrates multi-step RAG workflow
- **ChromaDB Vector Store**: Efficient document storage and retrieval
- **Cost Tracking**: Real-time tracking of OpenAI API costs and token usage
- **Structured Logging**: JSON-formatted logs with structlog
- **Docker Support**: Containerized deployment with docker-compose
- **CLI Interface**: Easy-to-use command-line interface
- **Report Export**: Save reports in text or markdown format
- **Observability**: Comprehensive logging and cost analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup:**
```bash
cd RAG
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

2. **Install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Ingest sample data:**
```bash
python scripts/ingest_sample_data.py
```

4. **Generate your first report:**
```bash
python main.py --query "What are the company's revenue trends?" --output --format markdown
```

## 🐳 Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker-compose build

# Or use the convenience script
./docker-run.sh build
```

### Common Docker commands:

```bash
# Ingest data
./docker-run.sh ingest ./data/raw_data

# Generate report
./docker-run.sh query "What are the revenue trends?"

# Show collection stats
./docker-run.sh stats

# Interactive shell
./docker-run.sh shell
```

## 📖 Usage

### CLI Commands

**Ingest Data:**
```bash
# Ingest a directory
python main.py --ingest-dir ./data/raw_data

# Ingest a single file
python main.py --ingest-file ./data/report.pdf
```

**Generate Reports:**
```bash
# Basic query
python main.py --query "Summarize Q4 financial results"

# Save as markdown
python main.py --query "What products does the company offer?" --output --format markdown

# Save as text
python main.py --query "Customer satisfaction metrics" --output --format text
```

**Utility Commands:**
```bash
# Show collection statistics
python main.py --stats

# Clear all data
python main.py --clear
```

### Programmatic Usage

```python
from src.agent.graph import ReportGenerationGraph
from src.retrieval.vector_store import VectorStore
from src.observability.logger import setup_logging

# Setup
setup_logging()
vector_store = VectorStore()

# Create graph
graph = ReportGenerationGraph(vector_store)

# Generate report
result = graph.generate_report("What are the key metrics?")

print(result["report"])
print(result["summary"])
print(f"Cost: ${result['total_cost']:.4f}")
```

## 🏛️ Architecture Details

### LangGraph Workflow

The report generation follows a state-machine workflow:

```
1. Retrieve Documents → 2. Build Context → 3. Generate Report → 4. Generate Summary
```

Each node in the graph:
- **retrieve_documents**: Searches vector store for relevant documents
- **build_context**: Combines retrieved documents into structured context
- **generate_report**: Uses LLM to create detailed report
- **generate_summary**: Generates executive summary

### Observability Layer

#### Logging
- Structured JSON logs for production
- Colored console output for development
- Automatic log rotation
- Application context in every log entry

#### Cost Tracking
- Real-time token counting
- Per-model cost calculation
- Session summaries
- Historical cost logs in JSON format

### Vector Store

ChromaDB configuration:
- Persistent storage on disk
- Configurable chunk size and overlap
- Similarity search with scores
- Support for metadata filtering

## 🔧 Configuration

All configuration is managed through environment variables in `.env`:

### Required Variables
```bash
OPENAI_API_KEY=your_key_here
```

### Optional Configuration
```bash
# Models
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Vector Store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Paths
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
REPORT_OUTPUT_DIR=./reports
LOG_DIR=./logs
```

## 📊 Supported Document Formats

- PDF (.pdf)
- Word Documents (.docx, .doc)
- Text Files (.txt, .md)
- More formats can be added in `utils/document_loader.py`

## 💰 Cost Management

The system tracks all OpenAI API calls and provides detailed cost breakdowns:

```bash
# Cost information is shown after each report generation
Total Cost: $0.0234
Input Tokens: 1,234
Output Tokens: 567

By Model:
  gpt-4-turbo-preview: $0.0234
```

Cost logs are saved to `logs/cost_tracking.json` for analysis.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_vector_store.py
```

## 📁 Project Structure

```
RAG/
├── src/                        # Source code
│   ├── agent/                  # LangGraph agent
│   │   ├── graph.py           # Workflow definition
│   │   ├── nodes.py           # Graph nodes
│   │   └── state.py           # State definition
│   ├── config/                 # Configuration
│   │   └── settings.py        # Settings with Pydantic
│   ├── data_ingestion/         # Data ingestion
│   │   └── ingestion_pipeline.py
│   ├── observability/          # Observability
│   │   ├── logger.py          # Structured logging
│   │   └── cost_tracker.py    # Cost tracking
│   ├── retrieval/              # Retrieval components
│   │   ├── embeddings.py      # Embeddings with tracking
│   │   └── vector_store.py    # ChromaDB wrapper
│   └── utils/                  # Utilities
│       ├── document_loader.py # Document loading
│       └── text_splitter.py   # Text chunking
├── scripts/                    # Utility scripts
│   ├── ingest_sample_data.py  # Sample data ingestion
│   └── example_queries.py     # Example queries
├── data/                       # Data directory
│   ├── raw_data/              # Source documents
│   └── chroma_db/             # Vector database
├── logs/                       # Log files
├── reports/                    # Generated reports
├── tests/                      # Test suite
├── main.py                     # CLI entry point
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker composition
├── docker-run.sh              # Docker convenience script
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                   # This file
```

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff src/ tests/

# Type checking
mypy src/
```

### Adding New Document Types

Edit `src/utils/document_loader.py`:

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".your_format": YourCustomLoader,
}
```

### Customizing the Workflow

Modify `src/agent/graph.py` to add new nodes or change the workflow:

```python
workflow.add_node("your_node", your_function)
workflow.add_edge("generate_report", "your_node")
```

## 🔒 Security Best Practices

- Never commit `.env` files
- Use environment variables for secrets
- Run containers as non-root user
- Keep dependencies updated
- Implement rate limiting for production

## 📈 Performance Optimization

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for your documents
- Use smaller embedding models for cost savings
- Implement caching for repeated queries
- Batch document ingestion for large datasets

## 🐛 Troubleshooting

### Vector store is empty
```bash
python main.py --stats
python scripts/ingest_sample_data.py
```

### OpenAI API errors
- Check your API key in `.env`
- Verify you have credits
- Check rate limits

### Docker issues
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```



