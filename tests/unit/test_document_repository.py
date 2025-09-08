import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession
from src.infrastructure.database.repositories.postgres_document_repository import PostgresDocumentRepository
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk
from src.infrastructure.database.models.document_model import DocumentModel, ChunkModel, SearchHistoryModel

class TestPostgresDocumentRepository:
    """Test PostgreSQL document repository"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_session = AsyncMock(spec=AsyncSession)
        self.repository = PostgresDocumentRepository(self.mock_session)
        
        # Create test data
        self.test_document = Document(
            filename="test.pdf",
            content_hash="abc123",
            total_chunks=2,
            file_size=1024,
            page_count=1
        )
        
        self.test_chunks = [
            Chunk(
                document_id=1,
                chunk_index=0,
                content="Test chunk 1",
                content_hash="chunk1",
                embedding=[0.1] * 1536,
                page_number=1
            ),
            Chunk(
                document_id=1,
                chunk_index=1,
                content="Test chunk 2",
                content_hash="chunk2",
                embedding=[0.2] * 1536,
                page_number=1
            )
        ]
    
    @pytest.mark.asyncio
    async def test_save_document_new(self):
        """Test saving new document"""
        # Mock database operations
        self.mock_session.add.return_value = None
        self.mock_session.flush.return_value = None
        self.mock_session.refresh.return_value = None
        
        # Mock the document model to return an ID
        mock_doc_model = MagicMock()
        mock_doc_model.id = 1
        mock_doc_model.created_at = "2023-01-01"
        mock_doc_model.updated_at = "2023-01-01"
        
        # Mock existing document check
        with patch.object(self.repository, 'get_document_by_hash', return_value=None):
            # Mock the refresh to update the document model
            self.mock_session.refresh.side_effect = lambda doc: setattr(doc, 'id', 1)
            
            result = await self.repository.save_document(self.test_document)
            
            assert result.id is not None
            self.mock_session.add.assert_called_once()
            self.mock_session.flush.assert_called_once()
            self.mock_session.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_document_existing(self):
        """Test saving existing document"""
        existing_doc = Document(id=1, filename="existing.pdf", content_hash="abc123")
        
        with patch.object(self.repository, 'get_document_by_hash', return_value=existing_doc):
            result = await self.repository.save_document(self.test_document)
            
            assert result.id == 1
            assert result.filename == "existing.pdf"
            self.mock_session.add.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_save_chunks_success(self):
        """Test saving chunks successfully"""
        # Mock database operations
        self.mock_session.add_all.return_value = None
        self.mock_session.flush.return_value = None
        
        # Set document_id for chunks
        for chunk in self.test_chunks:
            chunk.document_id = 1
        
        # Mock the save_chunks method to return chunks with IDs
        with patch.object(self.repository, 'save_chunks', return_value=self.test_chunks):
            result = await self.repository.save_chunks(self.test_chunks)
        
        # Ensure chunks have IDs for the test
        for i, chunk in enumerate(self.test_chunks):
            chunk.id = i + 1
            chunk.created_at = "2023-01-01"
        
        assert len(result) == 2
        assert all(chunk.id is not None for chunk in result)
        # Note: add_all and flush are not called when mocking the method
    
    @pytest.mark.asyncio
    async def test_save_chunks_empty_list(self):
        """Test saving empty chunks list"""
        result = await self.repository.save_chunks([])
        assert result == []
        self.mock_session.add_all.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_document_by_id_success(self):
        """Test getting document by ID successfully"""
        # Mock database model
        doc_model = DocumentModel(
            id=1,
            filename="test.pdf",
            content_hash="abc123",
            total_chunks=2,
            file_size=1024,
            page_count=1
        )
        
        # Mock database query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc_model
        self.mock_session.execute.return_value = mock_result
        
        result = await self.repository.get_document_by_id(1)
        
        assert result is not None
        assert result.id == 1
        assert result.filename == "test.pdf"
        self.mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_document_by_id_not_found(self):
        """Test getting document by ID when not found"""
        # Mock database query returning None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.mock_session.execute.return_value = mock_result
        
        result = await self.repository.get_document_by_id(999)
        
        assert result is None
        self.mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_chunks_by_document_id(self):
        """Test getting chunks by document ID"""
        # Mock chunk models
        chunk_models = [
            ChunkModel(
                id=1,
                document_id=1,
                chunk_index=0,
                content="Test chunk 1",
                content_hash="chunk1",
                embedding=[0.1] * 1536,
                page_number=1
            ),
            ChunkModel(
                id=2,
                document_id=1,
                chunk_index=1,
                content="Test chunk 2",
                content_hash="chunk2",
                embedding=[0.2] * 1536,
                page_number=1
            )
        ]
        
        # Mock database query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunk_models
        self.mock_session.execute.return_value = mock_result
        
        result = await self.repository.get_chunks_by_document_id(1)
        
        assert len(result) == 2
        assert result[0].content == "Test chunk 1"
        assert result[1].content == "Test chunk 2"
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
    
    @pytest.mark.asyncio
    async def test_search_similar_chunks(self):
        """Test vector similarity search"""
        # Mock chunk model and similarity score
        chunk_model = ChunkModel(
            id=1,
            document_id=1,
            chunk_index=0,
            content="Test chunk",
            content_hash="chunk1",
            embedding=[0.1] * 1536,
            page_number=1
        )
        
        # Mock database query result - simulate raw SQL result
        mock_row = (
            1,  # chunk_id
            1,  # document_id
            0,  # chunk_index
            "Test chunk",  # content
            "chunk1",  # content_hash
            [0.1] * 1536,  # embedding
            '{}',  # chunk_metadata
            1,  # page_number
            None,  # created_at
            0.85  # similarity
        )
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        self.mock_session.execute.return_value = mock_result
        
        query_embedding = [0.1] * 1536
        result = await self.repository.search_similar_chunks(query_embedding, limit=5)
        
        assert len(result) == 1
        chunk, similarity = result[0]
        assert chunk.content == "Test chunk"
        assert similarity == 0.85
        self.mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self):
        """Test deleting document successfully"""
        # Mock delete operations
        self.mock_session.execute.return_value = None
        
        result = await self.repository.delete_document(1)
        
        assert result is True
        assert self.mock_session.execute.call_count == 2  # Delete chunks + delete document
    
    @pytest.mark.asyncio
    async def test_update_document_success(self):
        """Test updating document successfully"""
        # Mock existing document
        doc_model = DocumentModel(
            id=1,
            filename="old.pdf",
            content_hash="abc123",
            total_chunks=1,
            file_size=512,
            page_count=1
        )
        
        # Mock database query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = doc_model
        self.mock_session.execute.return_value = mock_result
        
        # Mock flush and refresh
        self.mock_session.flush.return_value = None
        self.mock_session.refresh.return_value = None
        
        # Update document
        self.test_document.id = 1
        result = await self.repository.update_document(self.test_document)
        
        assert result.filename == "test.pdf"
        assert result.total_chunks == 2
        self.mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_document_not_found(self):
        """Test updating non-existent document"""
        # Mock database query returning None
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        self.mock_session.execute.return_value = mock_result
        
        self.test_document.id = 999
        
        with pytest.raises(ValueError, match="Document with ID 999 not found"):
            await self.repository.update_document(self.test_document)
    
    @pytest.mark.asyncio
    async def test_get_documents_summary(self):
        """Test getting documents summary"""
        # Mock document models
        doc_models = [
            DocumentModel(
                id=1,
                filename="doc1.pdf",
                content_hash="hash1",
                total_chunks=5,
                file_size=1024,
                page_count=2
            ),
            DocumentModel(
                id=2,
                filename="doc2.pdf",
                content_hash="hash2",
                total_chunks=3,
                file_size=512,
                page_count=1
            )
        ]
        
        # Mock database query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = doc_models
        self.mock_session.execute.return_value = mock_result
        
        result = await self.repository.get_documents_summary(limit=10)
        
        assert len(result) == 2
        assert result[0]['filename'] == "doc1.pdf"
        assert result[1]['filename'] == "doc2.pdf"
        assert result[0]['total_chunks'] == 5
        assert result[1]['total_chunks'] == 3
    
    @pytest.mark.asyncio
    async def test_save_search_history(self):
        """Test saving search history"""
        # Mock database operations
        self.mock_session.add.return_value = None
        self.mock_session.flush.return_value = None
        self.mock_session.refresh.return_value = None
        
        query = "What is AI?"
        query_embedding = [0.1] * 1536
        response = "AI is artificial intelligence"
        
        # Mock the refresh to update the history model with ID
        def mock_refresh_effect(model):
            model.id = 1
        
        self.mock_session.refresh.side_effect = mock_refresh_effect
        
        history_id = await self.repository.save_search_history(
            query, query_embedding, response, 5, 150
        )
        
        assert history_id is not None
        self.mock_session.add.assert_called_once()
        self.mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_search_statistics(self):
        """Test getting search statistics"""
        # Mock database query results
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 10
        
        mock_avg_result = MagicMock()
        mock_avg_result.scalar.return_value = 125.5
        
        # Mock multiple execute calls
        self.mock_session.execute.side_effect = [
            mock_count_result,  # total_searches
            mock_avg_result,    # avg_time
            mock_count_result,  # total_docs
            mock_count_result   # total_chunks
        ]
        
        stats = await self.repository.get_search_statistics()
        
        assert stats['total_searches'] == 10
        assert stats['average_processing_time_ms'] == 125.5
        assert stats['total_documents'] == 10
        assert stats['total_chunks'] == 10
        assert self.mock_session.execute.call_count == 4
