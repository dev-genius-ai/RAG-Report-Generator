# Quick Start Guide

Get up and running with the RAG Company Report Generator in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 2GB free disk space

## Installation Steps

### 1. Setup Environment

```bash
cd RAG

# Create .env file from template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 2. Install Dependencies

**Option A: Using pip (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using Docker**
```bash
./docker-run.sh build
```

### 3. Load Sample Data

```bash
# Without Docker
python scripts/ingest_sample_data.py

# With Docker
./docker-run.sh ingest /app/data/raw_data
```

Expected output:
```
🚀 Starting sample data ingestion...
📊 Current collection: company_data
   Documents before ingestion: 0
📥 Ingesting sample data...
✓ Successfully ingested 45 chunks
```

### 4. Generate Your First Report

```bash
# Without Docker
python main.py --query "What are ACME's revenue trends?"

# With Docker
./docker-run.sh query "What are ACME's revenue trends?"
```

## Example Queries

Try these example queries:

```bash
# Financial performance
python main.py --query "What are the Q4 2024 financial results?" --output --format markdown

# Product information
python main.py --query "What products does ACME offer?"

# Customer metrics
python main.py --query "Summarize customer satisfaction metrics"

# Strategic direction
python main.py --query "What are the company's future strategic initiatives?"
```

## Understanding the Output

A typical report includes:

1. **Executive Summary**: Key points in bullet format
2. **Detailed Report**: Comprehensive analysis
3. **Sources**: Documents used to generate the report
4. **Cost Summary**: API costs and token usage

Example:
```
EXECUTIVE SUMMARY
- ACME achieved $125M revenue in Q4 2024 (23% YoY growth)
- Cloud Services represent 52% of total revenue
- Operating margin improved to 28%

DETAILED REPORT
[Full analysis with specific data points...]

SOURCES
1. sample_company_data.txt
2. product_overview.txt

COST SUMMARY
Total Cost: $0.0234
Input Tokens: 1,234
Output Tokens: 567
```

## Common Commands

### Data Management

```bash
# Check collection status
python main.py --stats

# Clear all data
python main.py --clear

# Ingest your own documents
python main.py --ingest-dir ./path/to/your/documents
python main.py --ingest-file ./path/to/document.pdf
```

### Report Generation

```bash
# Basic query
python main.py --query "Your question here"

# Save as markdown
python main.py --query "Your question" --output --format markdown

# Save as text
python main.py --query "Your question" --output --format text
```

### Docker Commands

```bash
# Build image
./docker-run.sh build

# Ingest data
./docker-run.sh ingest /app/data/raw_data

# Generate report
./docker-run.sh query "Your question here"

# Check stats
./docker-run.sh stats

# Interactive shell
./docker-run.sh shell

# Stop containers
./docker-run.sh down
```

## Next Steps

1. **Add Your Own Data**
   - Place your documents in `data/raw_data/`
   - Supported formats: PDF, DOCX, TXT, MD
   - Run ingestion: `python main.py --ingest-dir ./data/raw_data`

2. **Customize Configuration**
   - Edit `.env` to change models, chunk sizes, etc.
   - See all options in `.env.example`

3. **Explore Advanced Features**
   - Review `ARCHITECTURE.md` for system design
   - Check `README.md` for detailed documentation
   - Examine example scripts in `scripts/`

4. **Integrate into Your Workflow**
   - Use programmatically (see README)
   - Build API endpoints
   - Schedule automated reports

## Troubleshooting

### "Vector store is empty"
```bash
python scripts/ingest_sample_data.py
```

### "OpenAI API Error"
- Verify your API key in `.env`
- Check you have available credits
- Ensure API key has correct permissions

### "Module not found"
```bash
pip install -r requirements.txt
```

### Docker issues
```bash
docker-compose down
docker-compose build --no-cache
```

## File Structure

```
RAG/
├── data/raw_data/          # Your documents go here
├── reports/                # Generated reports saved here
├── logs/                   # Application logs
├── main.py                 # Main CLI interface
└── scripts/                # Utility scripts
    ├── ingest_sample_data.py
    └── example_queries.py
```

## Support

- 📖 Full documentation: See `README.md`
- 🏗️ Architecture details: See `ARCHITECTURE.md`
- 🐛 Issues: Check logs in `logs/app.log`

## Performance Tips

- Adjust `CHUNK_SIZE` in `.env` for your document types
- Use `gpt-3.5-turbo` for faster, cheaper reports
- Batch queries to reduce overhead
- Monitor costs with built-in tracking

---

**You're ready to go!** Start generating insights from your company data. 🚀

