import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from src.config.database import get_async_session, init_database, close_database
from src.infrastructure.database.models.document_model import Base

class TestDatabaseConnection:
    """Test database connection and operations"""
    
    @pytest.mark.asyncio
    async def test_get_async_session(self):
        """Test async session creation"""
        session_gen = get_async_session()
        session = await anext(session_gen)
        
        assert isinstance(session, AsyncSession)
        # Note: AsyncSession doesn't have is_closed attribute in newer versions
        
        # Test session cleanup
        await session.close()
        with pytest.raises(StopAsyncIteration):
            await anext(session_gen)
    
    @patch('src.config.database.async_engine')
    @pytest.mark.asyncio
    async def test_init_database(self, mock_engine):
        """Test database initialization"""
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        await init_database()
        
        # Verify table creation
        mock_conn.run_sync.assert_called_once()
    
    @patch('src.config.database.async_engine')
    @pytest.mark.asyncio
    async def test_close_database(self, mock_engine):
        """Test database connection closure"""
        # Mock the dispose method to be async
        mock_engine.dispose = AsyncMock()
        
        await close_database()
        mock_engine.dispose.assert_called_once()

class TestDatabaseModels:
    """Test database models"""
    
    def test_document_model_creation(self):
        """Test DocumentModel creation"""
        from src.infrastructure.database.models.document_model import DocumentModel
        
        document = DocumentModel(
            filename="test.pdf",
            content_hash="abc123",
            total_chunks=5,
            file_size=1024,
            page_count=2
        )
        
        assert document.filename == "test.pdf"
        assert document.content_hash == "abc123"
        assert document.total_chunks == 5
        assert document.file_size == 1024
        assert document.page_count == 2
    
    def test_chunk_model_creation(self):
        """Test ChunkModel creation"""
        from src.infrastructure.database.models.document_model import ChunkModel
        import numpy as np
        
        # Create a mock embedding vector
        embedding = np.random.rand(1536).tolist()
        
        chunk = ChunkModel(
            document_id=1,
            chunk_index=0,
            content="Test chunk content",
            content_hash="chunk123",
            embedding=embedding,
            page_number=1
        )
        
        assert chunk.document_id == 1
        assert chunk.chunk_index == 0
        assert chunk.content == "Test chunk content"
        assert chunk.page_number == 1
    
    def test_chunk_metadata_property(self):
        """Test ChunkModel metadata property"""
        from src.infrastructure.database.models.document_model import ChunkModel
        
        chunk = ChunkModel()
        
        # Test setting metadata
        test_metadata = {"page": 1, "section": "intro"}
        chunk.metadata_dict = test_metadata
        
        # Test getting metadata
        assert chunk.metadata_dict == test_metadata
        
        # Test empty metadata
        chunk.metadata_dict = {}
        assert chunk.metadata_dict == {}
        
        # Test None metadata
        chunk.metadata_dict = None
        assert chunk.metadata_dict == {}
    
    def test_search_history_model_creation(self):
        """Test SearchHistoryModel creation"""
        from src.infrastructure.database.models.document_model import SearchHistoryModel
        import numpy as np
        
        query_embedding = np.random.rand(1536).tolist()
        
        history = SearchHistoryModel(
            query="What is AI?",
            query_embedding=query_embedding,
            response="AI is artificial intelligence",
            search_results_count=5,
            processing_time_ms=150
        )
        
        assert history.query == "What is AI?"
        assert history.response == "AI is artificial intelligence"
        assert history.search_results_count == 5
        assert history.processing_time_ms == 150
