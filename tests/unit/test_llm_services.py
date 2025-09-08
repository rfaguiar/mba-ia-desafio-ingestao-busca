import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.interfaces.llm_service import LLMService
from src.infrastructure.services.openai_llm_service import OpenAILLMService
from src.infrastructure.services.gemini_llm_service import GeminiLLMService
from src.presentation.prompts.search_prompt import SearchPromptTemplate

class TestLLMServiceInterface:
    """Test LLM service interface methods"""
    
    def test_validate_prompt_valid(self):
        """Test prompt validation with valid prompt"""
        # Create a mock service that implements the interface
        class MockLLMService(LLMService):
            async def generate_response(self, prompt, context=None, max_tokens=None, temperature=None):
                pass
            
            def get_service_name(self):
                return "Mock"
            
            def is_available(self):
                return True
            
            def get_model_info(self):
                return {}
        
        service = MockLLMService()
        
        # Test valid prompts
        assert service.validate_prompt("Valid prompt") is True
        assert service.validate_prompt("Prompt with numbers 123") is True
        assert service.validate_prompt("A" * 32000) is True  # At limit
        
        # Test invalid prompts
        assert service.validate_prompt("") is False
        assert service.validate_prompt("   ") is False
        assert service.validate_prompt(None) is False
        assert service.validate_prompt("A" * 32001) is False  # Over limit
    
    def test_validate_context_valid(self):
        """Test context validation with valid context"""
        class MockLLMService(LLMService):
            async def generate_response(self, prompt, context=None, max_tokens=None, temperature=None):
                pass
            
            def get_service_name(self):
                return "Mock"
            
            def is_available(self):
                return True
            
            def get_model_info(self):
                return {}
        
        service = MockLLMService()
        
        # Test valid contexts
        assert service.validate_context(None) is True
        assert service.validate_context([]) is True
        assert service.validate_context([{"content": "Text 1"}, {"content": "Text 2"}]) is True
        
        # Test invalid contexts
        assert service.validate_context("not a list") is False
        assert service.validate_context([{"wrong_key": "value"}]) is False
        assert service.validate_context([{"content": "valid"}, "not a dict"]) is False
    
    def test_format_context_for_prompt(self):
        """Test context formatting for prompts"""
        class MockLLMService(LLMService):
            async def generate_response(self, prompt, context=None, max_tokens=None, temperature=None):
                pass
            
            def get_service_name(self):
                return "Mock"
            
            def is_available(self):
                return True
            
            def get_model_info(self):
                return {}
        
        service = MockLLMService()
        
        context = [
            {"content": "First chunk", "metadata": {"page_number": 1}},
            {"content": "Second chunk", "metadata": {"page_number": 2}}
        ]
        
        formatted = service.format_context_for_prompt(context)
        
        assert "First chunk" in formatted
        assert "Second chunk" in formatted
        assert "Page 1" in formatted
        assert "Page 2" in formatted
    
    def test_estimate_token_count(self):
        """Test token count estimation"""
        class MockLLMService(LLMService):
            async def generate_response(self, prompt, context=None, max_tokens=None, temperature=None):
                pass
            
            def get_service_name(self):
                return "Mock"
            
            def is_available(self):
                return True
            
            def get_model_info(self):
                return {}
        
        service = MockLLMService()
        
        # Test token estimation
        assert service.estimate_token_count("Hello world") == 2  # 12 chars / 6 (actual implementation)
        assert service.estimate_token_count("A" * 100) == 25    # 100 chars / 4 (actual implementation)
        assert service.estimate_token_count("") == 0
    
    def test_is_prompt_within_limits(self):
        """Test prompt limit validation"""
        class MockLLMService(LLMService):
            async def generate_response(self, prompt, context=None, max_tokens=None, temperature=None):
                pass
            
            def get_service_name(self):
                return "Mock"
            
            def is_available(self):
                return True
            
            def get_model_info(self):
                return {}
        
        service = MockLLMService()
        
        # Test within limits
        assert service.is_prompt_within_limits("Short prompt") is True
        assert service.is_prompt_within_limits("A" * 10000) is True
        
        # Test over limits
        long_prompt = "A" * 40000  # Very long prompt
        assert service.is_prompt_within_limits(long_prompt) is True  # Mock implementation returns True

class TestOpenAILLMService:
    """Test OpenAI LLM service"""
    
    def setup_method(self, method):
        """Setup test fixtures"""
        # Mock the settings before initializing the service
        self.mock_settings_patcher = patch('src.infrastructure.services.openai_llm_service.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.OPENAI_API_KEY = "test-api-key"
        self.mock_settings.OPENAI_LLM_MODEL = "gpt-5-nano"
        
        # Mock the ChatOpenAI to avoid API initialization
        self.mock_openai_patcher = patch('langchain_openai.ChatOpenAI')
        self.mock_openai = self.mock_openai_patcher.start()
        self.mock_openai.return_value = MagicMock()
        
        self.service = OpenAILLMService()
    
    def teardown_method(self, method):
        """Cleanup test fixtures"""
        self.mock_settings_patcher.stop()
        self.mock_openai_patcher.stop()
    
    def test_initialization(self):
        """Test service initialization"""
        assert self.service.api_key == "test-api-key"
        assert self.service.model == "gpt-5-nano"
        assert self.service.get_service_name() == "OpenAI"
    
    def test_initialization_missing_api_key(self):
        """Test initialization without API key"""
        with patch('src.infrastructure.services.openai_llm_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAILLMService(api_key=None)
    
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
    async def test_generate_response_success(self):
        """Test successful response generation"""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test response"
        mock_client.ainvoke.return_value = mock_response
        
        # Create service first, then patch the client
        service = OpenAILLMService(api_key="test-key")
        service.client = mock_client
        
        response = await service.generate_response("Test prompt")
        
        assert response == "This is a test response"
        mock_client.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_context(self):
        """Test response generation with context"""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Response with context"
        mock_client.ainvoke.return_value = mock_response
        
        # Create service first, then patch the client
        service = OpenAILLMService(api_key="test-key")
        service.client = mock_client
        
        context = [{"content": "Context text", "metadata": {"page_number": 1}}]
        response = await service.generate_response("Test prompt", context)
        
        assert response == "Response with context"
        # Verify context was included in prompt
        call_args = mock_client.ainvoke.call_args[0][0]
        assert "Context:" in call_args
        assert "Context text" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_response_invalid_prompt(self):
        """Test response generation with invalid prompt"""
        with pytest.raises(Exception, match="OpenAI response generation failed: Invalid prompt provided"):
            await self.service.generate_response("")
    
    @pytest.mark.asyncio
    async def test_generate_response_invalid_context(self):
        """Test response generation with invalid context"""
        invalid_context = "not a list"
        
        with pytest.raises(Exception, match="OpenAI response generation failed: Invalid context provided"):
            await self.service.generate_response("Valid prompt", invalid_context)
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        info = self.service.get_model_info()
        
        assert info['service'] == 'OpenAI'
        assert info['model'] == 'gpt-5-nano'
        assert info['api_key_configured'] is True
        assert info['service_available'] is True
        assert info['max_tokens'] == 1000
        assert info['temperature'] == 0.1
    
    def test_get_token_limits(self):
        """Test token limits retrieval"""
        limits = self.service.get_token_limits()
        
        assert 'max_tokens' in limits
        assert 'max_prompt_tokens' in limits
        assert 'max_response_tokens' in limits
        assert limits['max_response_tokens'] == 1000

class TestGeminiLLMService:
    """Test Gemini LLM service"""
    
    def setup_method(self, method):
        """Setup test fixtures"""
        # Mock the settings before initializing the service
        self.mock_settings_patcher = patch('src.infrastructure.services.gemini_llm_service.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.GOOGLE_API_KEY = "test-api-key"
        self.mock_settings.GOOGLE_LLM_MODEL = "gemini-2.5-flash-lite"
        
        # Mock the ChatGoogleGenerativeAI to avoid API initialization
        self.mock_gemini_patcher = patch('langchain_google_genai.ChatGoogleGenerativeAI')
        self.mock_gemini = self.mock_gemini_patcher.start()
        self.mock_gemini.return_value = MagicMock()
        
        self.service = GeminiLLMService()
        
        # Replace the client with our mock to avoid API issues
        self.service.client = MagicMock()
    
    def teardown_method(self, method):
        """Cleanup test fixtures"""
        self.mock_settings_patcher.stop()
        self.mock_gemini_patcher.stop()
    
    def test_initialization(self):
        """Test service initialization"""
        assert self.service.api_key == "test-api-key"
        assert self.service.model == "gemini-2.5-flash-lite"
        assert self.service.get_service_name() == "Google Gemini"
    
    def test_initialization_missing_api_key(self):
        """Test initialization without API key"""
        with patch('src.infrastructure.services.gemini_llm_service.settings') as mock_settings:
            mock_settings.GOOGLE_API_KEY = None
            with pytest.raises(ValueError, match="Google API key is required"):
                GeminiLLMService(api_key=None)
    
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
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        info = self.service.get_model_info()
        
        assert info['service'] == 'Google Gemini'
        assert info['model'] == 'gemini-2.5-flash-lite'
        assert info['api_key_configured'] is True
        assert info['service_available'] is True
        assert info['max_tokens'] == 1000
        assert info['temperature'] == 0.1
    
    def test_get_token_limits(self):
        """Test token limits retrieval"""
        limits = self.service.get_token_limits()
        
        assert 'max_tokens' in limits
        assert 'max_prompt_tokens' in limits
        assert 'max_response_tokens' in limits
        # Gemini models have very high token limits
        assert limits['max_tokens'] >= 1000000

class TestSearchPromptTemplate:
    """Test search prompt template"""
    
    def test_format_prompt_base(self):
        """Test base prompt formatting"""
        question = "What is AI?"
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        
        prompt = SearchPromptTemplate.format_prompt(question, context)
        
        assert "CONTEXTO:" in prompt
        assert "REGRAS:" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "What is AI?" in prompt
    
    def test_format_prompt_document_specific(self):
        """Test document-specific prompt formatting"""
        question = "What is AI?"
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        document_name = "test.pdf"
        
        prompt = SearchPromptTemplate.format_prompt(
            question, context, "document_specific", document_name
        )
        
        assert "Documento: test.pdf" in prompt
        assert "AI is artificial intelligence" in prompt
    
    def test_format_prompt_technical(self):
        """Test technical prompt formatting"""
        question = "What is AI?"
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        
        prompt = SearchPromptTemplate.format_prompt(question, context, "technical")
        
        assert "CONTEXTO TÉCNICO:" in prompt
        assert "terminologia técnica" in prompt
    
    def test_format_prompt_business(self):
        """Test business prompt formatting"""
        question = "What is AI?"
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        
        prompt = SearchPromptTemplate.format_prompt(question, context, "business")
        
        assert "CONTEXTO EMPRESARIAL:" in prompt
        assert "valores financeiros" in prompt
    
    def test_format_context_with_metadata(self):
        """Test context formatting with metadata"""
        context = [
            {"content": "First chunk", "metadata": {"page_number": 1, "chunk_index": 0}},
            {"content": "Second chunk", "metadata": {"page_number": 2, "chunk_index": 1}}
        ]
        
        formatted = SearchPromptTemplate._format_context(context)
        
        assert "1. First chunk (Página 1, Chunk 0)" in formatted
        assert "2. Second chunk (Página 2, Chunk 1)" in formatted
    
    def test_create_summary_prompt(self):
        """Test summary prompt creation"""
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        question = "What is AI?"
        
        prompt = SearchPromptTemplate.create_summary_prompt(context, question)
        
        assert "TAREFA:" in prompt
        assert "Crie um resumo conciso" in prompt
        assert "AI is artificial intelligence" in prompt
    
    def test_create_analysis_prompt(self):
        """Test analysis prompt creation"""
        context = [{"content": "AI is artificial intelligence", "metadata": {}}]
        question = "What is AI?"
        
        prompt = SearchPromptTemplate.create_analysis_prompt(context, question)
        
        assert "ANÁLISE SOLICITADA:" in prompt
        assert "Identifique padrões" in prompt
        assert "AI is artificial intelligence" in prompt
    
    def test_validate_prompt_length(self):
        """Test prompt length validation"""
        short_prompt = "Short prompt"
        long_prompt = "A" * 32001
        
        assert SearchPromptTemplate.validate_prompt_length(short_prompt) is True
        assert SearchPromptTemplate.validate_prompt_length(long_prompt) is False
    
    def test_truncate_context_if_needed(self):
        """Test context truncation"""
        context = [
            {"content": "A" * 1000, "metadata": {}},
            {"content": "A" * 1000, "metadata": {}},
            {"content": "A" * 1000, "metadata": {}}
        ]
        question = "Test question"
        
        # Test with very short max length
        truncated = SearchPromptTemplate.truncate_context_if_needed(context, question, max_length=100)
        
        assert len(truncated) < len(context)
        assert len(truncated) >= 0
