"""
Configuration settings for the RAG application.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Settings
    api_title: str = Field(
        default="RAG Vector Search API", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")

    # Qdrant Settings
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection_name: str = Field(
        default="docs_qdrant", description="Qdrant collection name")
    qdrant_vector_size: int = Field(
        default=384, description="Vector embedding dimension")
    qdrant_distance_metric: str = Field(
        default="Cosine", description="Distance metric (Cosine, Dot, Euclid)")

    # PostgreSQL + pgvector Settings
    postgres_host: str = Field(
        default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(
        default="pguser", description="PostgreSQL username")
    postgres_password: str = Field(
        default="pgpass", description="PostgreSQL password")
    postgres_database: str = Field(
        default="vectordb", description="PostgreSQL database name")
    postgres_table_name: str = Field(
        default="docs", description="PostgreSQL table name")

    # Embedding Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_device: str = Field(
        default="cpu", description="Device for embedding model (cpu/cuda)")

    # Text Processing Settings
    default_chunk_size: int = Field(
        default=500, description="Default text chunk size")
    default_chunk_overlap: int = Field(
        default=50, description="Default chunk overlap")
    max_content_length: int = Field(
        default=10000, description="Maximum content length for processing")

    # RAG Settings
    default_system_prompt: str = Field(
        default=(
            "Eres un asistente especializado en análisis de datos y bases de datos avanzadas. "
            "Utiliza el contexto proporcionado para responder preguntas de manera precisa y detallada. "
            "Si no tienes información suficiente en el contexto, indícalo claramente."
        ),
        description="Default system prompt for RAG"
    )
    max_context_tokens: int = Field(
        default=4000, description="Maximum tokens for context")
    temperature: float = Field(
        default=0.1, description="Temperature for text generation")

    # LLM Settings (for future integration)
    llm_provider: str = Field(
        default="openai", description="LLM provider (openai, anthropic, etc.)")
    llm_model: str = Field(default="gpt-3.5-turbo",
                           description="LLM model name")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key")
    llm_base_url: Optional[str] = Field(
        default=None, description="Custom LLM base URL")

    # File Processing Settings
    supported_file_types: list = Field(
        default=[".txt", ".md", ".pdf", ".docx"],
        description="Supported file types for ingestion"
    )
    data_directory: str = Field(
        default="./data/raw", description="Directory containing raw data files")

    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )

    # Performance Settings
    batch_size: int = Field(
        default=10, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")
    connection_pool_size: int = Field(
        default=10, description="Database connection pool size")

    # Security Settings
    allowed_origins: list = Field(
        default=["*"], description="CORS allowed origins")
    api_key: Optional[str] = Field(
        default=None, description="API key for authentication")

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def postgres_async_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    class Config:
        """Pydantic config for environment variable loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

        # Environment variable prefixes
        env_prefix = ""

    def get_embedding_config(self) -> dict:
        """Get embedding model configuration."""
        return {
            "model_name": self.embedding_model,
            "device": self.embedding_device,
            "dimension": self.qdrant_vector_size
        }

    def get_qdrant_config(self) -> dict:
        """Get Qdrant configuration."""
        return {
            "host": self.qdrant_host,
            "port": self.qdrant_port,
            "collection_name": self.qdrant_collection_name,
            "vector_size": self.qdrant_vector_size,
            "distance": self.qdrant_distance_metric
        }

    def get_postgres_config(self) -> dict:
        """Get PostgreSQL configuration."""
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "password": self.postgres_password,
            "database": self.postgres_database,
            "table_name": self.postgres_table_name
        }

    def get_processing_config(self) -> dict:
        """Get text processing configuration."""
        return {
            "chunk_size": self.default_chunk_size,
            "chunk_overlap": self.default_chunk_overlap,
            "max_content_length": self.max_content_length,
            "supported_types": self.supported_file_types
        }
