from .settings import settings
import logging
from typing import List

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validate configuration settings"""

    @staticmethod
    def validate_required_settings() -> List[str]:
        """Validate required configuration settings"""
        errors = []

        # Check if at least one AI service is configured
        if not settings.OPENAI_API_KEY and not settings.GOOGLE_API_KEY:
            errors.append("Either OPENAI_API_KEY or GOOGLE_API_KEY must be configured")

        # Validate database settings
        if not settings.POSTGRES_HOST:
            errors.append("POSTGRES_HOST is required")

        if not settings.POSTGRES_USER:
            errors.append("POSTGRES_USER is required")

        if not settings.POSTGRES_PASSWORD:
            errors.append("POSTGRES_PASSWORD is required")

        if not settings.POSTGRES_DB:
            errors.append("POSTGRES_DB is required")

        # Validate processing settings
        if settings.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be greater than 0")

        if settings.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")

        if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        if settings.SEARCH_RESULTS_LIMIT <= 0:
            errors.append("SEARCH_RESULTS_LIMIT must be greater than 0")

        return errors

    @staticmethod
    def validate_and_log():
        """Validate configuration and log results"""
        errors = ConfigurationValidator.validate_required_settings()

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info("Configuration validation passed")

        # Log configuration summary
        logger.info("Configuration Summary:")
        logger.info(
            f"  Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        logger.info(
            f"  OpenAI: {'Configured' if settings.OPENAI_API_KEY else 'Not configured'}"
        )
        logger.info(
            f"  Gemini: {'Configured' if settings.GOOGLE_API_KEY else 'Not configured'}"
        )
        logger.info(f"  Chunk Size: {settings.CHUNK_SIZE}")
        logger.info(f"  Chunk Overlap: {settings.CHUNK_OVERLAP}")
        logger.info(f"  Search Results Limit: {settings.SEARCH_RESULTS_LIMIT}")
