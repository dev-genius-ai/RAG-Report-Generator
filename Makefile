# Makefile for RAG Company Report Generator

.PHONY: help install test lint format clean docker-build docker-run ingest query

# Default target
help:
	@echo "RAG Company Report Generator - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          - Install dependencies"
	@echo "  make setup            - Setup environment and ingest sample data"
	@echo ""
	@echo "Development:"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linters"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean generated files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo ""
	@echo "Application:"
	@echo "  make ingest           - Ingest sample data"
	@echo "  make query Q='text'   - Generate report for query"
	@echo "  make stats            - Show collection statistics"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

setup: install
	cp .env.example .env
	@echo "Please edit .env and add your OPENAI_API_KEY"
	@echo "Then run: make ingest"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf dist build

clean-data:
	rm -rf data/chroma_db/*
	@echo "Vector database cleared"

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose run --rm rag-app /bin/bash

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Application commands
ingest:
	python scripts/ingest_sample_data.py

query:
	python main.py --query "$(Q)" --output --format markdown

stats:
	python main.py --stats

clear:
	python main.py --clear

# Example queries
examples:
	python scripts/example_queries.py

# Development server (if API is implemented)
serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

