"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://article_mind:article_mind@localhost:5432/article_mind"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://192.168.1.9:5175"
    cors_allow_all: bool = False

    # App
    debug: bool = False
    log_level: str = "INFO"
    app_name: str = "Article Mind Service"
    app_version: str = "0.1.0"

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
