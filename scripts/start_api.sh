#!/bin/bash
# Start the FastAPI server

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG API Server...${NC}"
echo ""
echo "API will be available at:"
echo "  • Main API: http://localhost:8000"
echo "  • Interactive Docs (Swagger): http://localhost:8000/docs"
echo "  • Alternative Docs (ReDoc): http://localhost:8000/redoc"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

