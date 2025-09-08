import pytest
import os
from unittest.mock import patch
from src.config.settings import Settings
from src.config.validator import ConfigurationValidator

class TestSettings:
    """Test configuration settings"""
    
    def test_default_values(self):
        """Test default configuration values"""
        settings = Settings()
        
        assert settings.POSTGRES_HOST == "localhost"
        assert settings.POSTGRES_PORT == 5432
        assert settings.POSTGRES_USER == "postgres"
        assert settings.POSTGRES_PASSWORD == "postgres"
        assert settings.POSTGRES_DB == "rag"
        assert settings.CHUNK_SIZE == 1000
        assert settings.CHUNK_OVERLAP == 150
        assert settings.SEARCH_RESULTS_LIMIT == 10
    
    def test_database_url_generation(self):
        """Test database URL generation"""
        settings = Settings()
        
        expected_sync_url = "postgresql://postgres:postgres@localhost:5432/rag"
        expected_async_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag"
        
        assert settings.database_url == expected_sync_url
        assert settings.async_database_url == expected_async_url
    
    @patch.dict(os.environ, {
        'POSTGRES_HOST': 'custom-host',
        'POSTGRES_PORT': '5433',
        'OPENAI_API_KEY': 'test-key'
    })
    def test_environment_override(self):
        """Test environment variable override"""
        settings = Settings()
        
        assert settings.POSTGRES_HOST == "custom-host"
        assert settings.POSTGRES_PORT == 5433
        assert settings.OPENAI_API_KEY == "test-key"

class TestConfigurationValidator:
    """Test configuration validation"""
    
    def test_valid_configuration(self):
        """Test valid configuration"""
        with patch('src.config.settings.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.POSTGRES_HOST = "localhost"
            mock_settings.POSTGRES_USER = "postgres"
            mock_settings.POSTGRES_PASSWORD = "postgres"
            mock_settings.POSTGRES_DB = "rag"
            mock_settings.CHUNK_SIZE = 1000
            mock_settings.CHUNK_OVERLAP = 150
            mock_settings.SEARCH_RESULTS_LIMIT = 10
            
            errors = ConfigurationValidator.validate_required_settings()
            assert len(errors) == 0
    
    def test_missing_ai_service(self):
        """Test missing AI service configuration"""
        with patch('src.config.validator.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            mock_settings.GOOGLE_API_KEY = None
            mock_settings.POSTGRES_HOST = "localhost"
            mock_settings.POSTGRES_USER = "postgres"
            mock_settings.POSTGRES_PASSWORD = "postgres"
            mock_settings.POSTGRES_DB = "rag"
            mock_settings.CHUNK_SIZE = 1000
            mock_settings.CHUNK_OVERLAP = 150
            mock_settings.SEARCH_RESULTS_LIMIT = 10
            
            errors = ConfigurationValidator.validate_required_settings()
            assert len(errors) == 1
            assert "Either OPENAI_API_KEY or GOOGLE_API_KEY must be configured" in errors[0]
    
    def test_invalid_chunk_size(self):
        """Test invalid chunk size"""
        with patch('src.config.validator.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.POSTGRES_HOST = "localhost"
            mock_settings.POSTGRES_USER = "postgres"
            mock_settings.POSTGRES_PASSWORD = "postgres"
            mock_settings.POSTGRES_DB = "rag"
            mock_settings.CHUNK_SIZE = 0
            mock_settings.CHUNK_OVERLAP = 150
            mock_settings.SEARCH_RESULTS_LIMIT = 10
            
            errors = ConfigurationValidator.validate_required_settings()
            assert len(errors) == 2
            assert "CHUNK_SIZE must be greater than 0" in errors
            assert "CHUNK_OVERLAP must be less than CHUNK_SIZE" in errors
    
    def test_invalid_chunk_overlap(self):
        """Test invalid chunk overlap"""
        with patch('src.config.validator.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.POSTGRES_HOST = "localhost"
            mock_settings.POSTGRES_USER = "postgres"
            mock_settings.POSTGRES_PASSWORD = "postgres"
            mock_settings.POSTGRES_DB = "rag"
            mock_settings.CHUNK_SIZE = 1000
            mock_settings.CHUNK_OVERLAP = 1001
            mock_settings.SEARCH_RESULTS_LIMIT = 10
            
            errors = ConfigurationValidator.validate_required_settings()
            assert len(errors) == 1
            assert "CHUNK_OVERLAP must be less than CHUNK_SIZE" in errors[0]
