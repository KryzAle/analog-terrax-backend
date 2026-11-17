from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Analog Terrax API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "preview", "production", "test"] = "development"
    API_V1_STR: str = "/api/v1"

    # Cross-Origin Resource Sharing (CORS)
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://analog-terrax.vercel.app",
            "https://analog-terrax-frontend.vercel.app",
        ]
    )
    CORS_ORIGIN_REGEX: str = r"^https://analog-terrax-frontend[a-z0-9-]*\.vercel\.app$"

    # Per-IP requests/minute on compute-heavy endpoints (0 disables)
    RATE_LIMIT_PER_MINUTE: int = 60

    # Vercel Web Analytics
    ENABLE_WEB_ANALYTICS: bool = True

    # Outbound HTTP & NASA API Settings
    UPSTREAM_TIMEOUT_SECONDS: float = 15.0
    NASA_API_KEY: str = "DEMO_KEY"
    NASA_API_BASE_URL: str = "https://api.nasa.gov"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
