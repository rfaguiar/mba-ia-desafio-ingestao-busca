from pydantic_settings import BaseSettings
from typing import Optional
import os
from .ai_service_selector import get_ai_service, get_service_info


class Settings(BaseSettings):
    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "rag"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-5-nano"

    # Google Gemini Configuration (Alternative)
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_EMBEDDING_MODEL: str = "models/embedding-001"
    GOOGLE_LLM_MODEL: str = "gemini-2.5-flash-lite"

    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    SEARCH_RESULTS_LIMIT: int = 10

    # PDF Configuration
    PDF_PATH: str = "document.pdf"

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @property
    def database_url(self) -> str:
        """Generate database connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def async_database_url(self) -> str:
        """Generate async database connection URL"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def selected_ai_service(self) -> Optional[str]:
        """Get the currently selected AI service"""
        return get_ai_service()
    
    @property
    def ai_service_info(self) -> dict:
        """Get information about AI service configuration"""
        return get_service_info()

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
