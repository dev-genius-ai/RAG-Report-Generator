"""
Application settings and configuration management.
Uses Pydantic for validation and environment variable loading.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI Model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="OpenAI Embedding Model"
    )

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="company_data", description="ChromaDB collection name"
    )

    # Application Configuration
    app_name: str = Field(default="RAG Company Report Generator", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment")

    # Vector Store Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size for splitting")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    top_k_results: int = Field(default=5, description="Number of top results to retrieve")

    # Report Configuration
    max_report_length: int = Field(default=2000, description="Maximum report length in tokens")
    report_output_dir: str = Field(default="./reports", description="Report output directory")

    # Cost Tracking
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    cost_log_file: str = Field(
        default="./logs/cost_tracking.json", description="Cost tracking log file"
    )

    # Logging Configuration
    log_dir: str = Field(default="./logs", description="Log directory")
    log_file: str = Field(default="app.log", description="Log file name")
    log_rotation_size_mb: int = Field(default=10, description="Log rotation size in MB")
    log_backup_count: int = Field(default=5, description="Number of log backups to keep")

    def get_absolute_path(self, path: str) -> Path:
        """Convert relative path to absolute path."""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            return Path.cwd() / path_obj
        return path_obj

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.chroma_persist_directory,
            self.report_output_dir,
            self.log_dir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings

