"""Application configuration using Pydantic Settings."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://article_mind:article_mind@localhost:5432/article_mind"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = (
        "http://localhost:13000,http://localhost:13001,http://localhost:13002,http://192.168.1.9:13002"
    )
    cors_allow_all: bool = False

    # App
    debug: bool = False
    log_level: str = "INFO"
    sqlalchemy_log_level: str = "WARNING"  # SQLAlchemy-specific log level (INFO shows all SQL)
    app_name: str = "Article Mind Service"
    app_version: str = "0.1.0"

    # File Upload
    upload_base_path: str = "data/uploads"
    max_upload_size_mb: int = 50

    # Content Extraction
    extraction_timeout_seconds: int = 30
    extraction_max_retries: int = 3
    extraction_user_agent: str = "ArticleMind/1.0 (Content Extraction Bot)"
    playwright_headless: bool = True
    extraction_max_content_size_mb: int = 50

    # Embedding Provider
    embedding_provider: Literal["openai", "ollama"] = "openai"

    # OpenAI
    openai_api_key: str | None = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # ChromaDB
    chromadb_path: str = "./data/chromadb"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    anthropic_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 2048

    # RAG Configuration
    rag_context_chunks: int = 5

    # Search Configuration
    search_top_k: int = 10
    search_dense_weight: float = 0.7
    search_sparse_weight: float = 0.3
    search_rrf_k: int = 60
    search_rerank_enabled: bool = False
    search_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    search_rerank_top_k: int = 20
    search_max_query_length: int = 1000
    search_timeout_seconds: int = 30
    chroma_persist_directory: str = "./data/chromadb"
    chroma_collection_name: str = "article_chunks"

    @property
    def embedding_dimensions(self) -> int:
        """Get dimensions based on configured provider.

        Returns:
            1536 for OpenAI text-embedding-3-small
            1024 for Ollama nomic-embed-text
        """
        if self.embedding_provider == "openai":
            return 1536
        return 1024  # ollama nomic-embed-text

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string.

        Returns:
            List of allowed origins, or ["*"] if cors_allow_all is True.
        """
        if self.cors_allow_all:
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
