import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.infrastructure.services.openai_embedding_service import OpenAIEmbeddingService
from src.infrastructure.services.gemini_embedding_service import GeminiEmbeddingService


class TestOpenAIEmbeddingService:
    """Test OpenAI embedding service"""
    
    def setup_method(self, method):
        """Setup test fixtures"""
        # Mock the settings before initializing the service
        self.mock_settings_patcher = patch('src.infrastructure.services.openai_embedding_service.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.OPENAI_API_KEY = "test-api-key"
        self.mock_settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        # Mock the OpenAIEmbeddings to avoid API initialization
        self.mock_openai_patcher = patch('langchain_openai.OpenAIEmbeddings')
        self.mock_openai = self.mock_openai_patcher.start()
        self.mock_openai.return_value = MagicMock()
        
        # Create service after mocking
        self.service = OpenAIEmbeddingService()
        
        # Replace the client with our mock to avoid API issues
        self.service.client = MagicMock()
    
    def teardown_method(self, method):
        """Cleanup test fixtures"""
        self.mock_settings_patcher.stop()
        self.mock_openai_patcher.stop()
    
    def test_initialization(self):
        """Test service initialization"""
        assert self.service.api_key == "test-api-key"
        assert self.service.model == "text-embedding-3-small"
        assert self.service.get_service_name() == "OpenAI"
        assert self.service.get_embedding_dimension() == 1536
    
    def test_initialization_missing_api_key(self):
        """Test initialization without API key"""
        with patch('src.infrastructure.services.openai_embedding_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIEmbeddingService(api_key=None)
    
    def test_is_available(self):
        """Test service availability check"""
        # Test with valid API key
        assert self.service.is_available() is True
        
        # Test with placeholder API key
        self.service.api_key = "your_openai_api_key_here"
        assert self.service.is_available() is False
        
        # Test with None API key
        self.service.api_key = None
        assert self.service.is_available() is False
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding_success(self):
        """Test successful single embedding generation"""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_client.aembed_query.return_value = [0.1] * 1536
        
        # Create service first, then patch the client
        service = OpenAIEmbeddingService(api_key="test-key")
        service.client = mock_client
        
        embedding = await service.generate_single_embedding("Test text")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        mock_client.aembed_query.assert_called_once_with("Test text")
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful batch embedding generation"""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_client.aembed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
        
        # Create service first, then patch the client
        service = OpenAIEmbeddingService(api_key="test-key")
        service.client = mock_client
        
        texts = ["Text 1", "Text 2"]
        embeddings = await service.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)
        mock_client.aembed_documents.assert_called_once_with(texts)
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        info = self.service.get_model_info()
        
        assert info['service'] == 'OpenAI'
        assert info['model'] == 'text-embedding-3-small'
        assert info['dimension'] == 1536
        assert info['api_key_configured'] is True
        assert info['service_available'] is True


class TestGeminiEmbeddingService:
    """Test Gemini embedding service"""
    
    def setup_method(self, method):
        """Setup test fixtures"""
        # Mock the settings before initializing the service
        self.mock_settings_patcher = patch('src.infrastructure.services.gemini_embedding_service.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.GOOGLE_API_KEY = "test-api-key"
        self.mock_settings.GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
        
        # Mock the GoogleGenerativeAIEmbeddings class completely to avoid gRPC initialization
        self.mock_gemini_patcher = patch('src.infrastructure.services.gemini_embedding_service.GoogleGenerativeAIEmbeddings')
        self.mock_gemini_class = self.mock_gemini_patcher.start()
        
        # Create a mock instance that will be returned by the class constructor
        self.mock_client = MagicMock()
        self.mock_gemini_class.return_value = self.mock_client
        
        # Create service after mocking
        self.service = GeminiEmbeddingService()
    
    def teardown_method(self, method):
        """Cleanup test fixtures"""
        self.mock_settings_patcher.stop()
        self.mock_gemini_patcher.stop()
    
    def test_initialization(self):
        """Test service initialization"""
        assert self.service.api_key == "test-api-key"
        assert self.service.model == "models/embedding-001"
        assert self.service.get_service_name() == "Google Gemini"
        assert self.service.get_embedding_dimension() == 768
    
    def test_initialization_missing_api_key(self):
        """Test initialization without API key"""
        with patch('src.infrastructure.services.gemini_embedding_service.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            with pytest.raises(ValueError, match="Google API key is required"):
                GeminiEmbeddingService(api_key=None)
    
    def test_is_available(self):
        """Test service availability check"""
        # Test with valid API key
        assert self.service.is_available() is True
        
        # Test with placeholder API key
        self.service.api_key = "your_google_api_key_here"
        assert self.service.is_available() is False
        
        # Test with None API key
        self.service.api_key = None
        assert self.service.is_available() is False
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding_success(self):
        """Test successful single embedding generation"""
        # Mock Gemini client
        mock_client = AsyncMock()
        mock_client.aembed_query.return_value = [0.1] * 768
        
        # Create service first, then patch the client
        service = GeminiEmbeddingService(api_key="test-key")
        service.client = mock_client
        
        embedding = await service.generate_single_embedding("Test text")
        
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
        mock_client.aembed_query.assert_called_once_with("Test text")
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self):
        """Test successful batch embedding generation"""
        # Mock Gemini client
        mock_client = AsyncMock()
        mock_client.aembed_documents.return_value = [[0.1] * 768, [0.2] * 768]
        
        # Create service first, then patch the client
        service = GeminiEmbeddingService(api_key="test-key")
        service.client = mock_client
        
        texts = ["Text 1", "Text 2"]
        embeddings = await service.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 768 for emb in embeddings)
        mock_client.aembed_documents.assert_called_once_with(texts)
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        info = self.service.get_model_info()
        
        assert info['service'] == 'Google Gemini'
        assert info['model'] == 'models/embedding-001'
        assert info['dimension'] == 768
        assert info['api_key_configured'] is True
        assert info['service_available'] is True
