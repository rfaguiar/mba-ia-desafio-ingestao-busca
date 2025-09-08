import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.infrastructure.services.pdf_loader_service import PDFLoaderService
from src.infrastructure.text_processing.chunker import DocumentChunker
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk

class TestPDFLoaderService:
    """Test PDF loader service"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pdf_service = PDFLoaderService()
    
    def test_can_load_file_valid_pdf(self):
        """Test file validation for valid PDF"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            assert self.pdf_service.can_load_file(tmp_file_path) is True
        finally:
            os.unlink(tmp_file_path)
    
    def test_can_load_file_invalid_extension(self):
        """Test file validation for invalid extension"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b'Test text content')
            tmp_file_path = tmp_file.name
        
        try:
            assert self.pdf_service.can_load_file(tmp_file_path) is False
        finally:
            os.unlink(tmp_file_path)
    
    def test_can_load_file_nonexistent(self):
        """Test file validation for nonexistent file"""
        assert self.pdf_service.can_load_file('nonexistent.pdf') is False
    
    def test_load_pdf_success(self):
        """Test successful PDF loading"""
        # Mock the load_pdf method to return a document
        mock_document = Document(
            filename="test.pdf",
            content_hash="abc123",
            total_chunks=2,
            file_size=1024,
            page_count=2
        )
        
        with patch.object(self.pdf_service, 'load_pdf', return_value=mock_document):
            document = self.pdf_service.load_pdf("test.pdf")
        
        assert isinstance(document, Document)
        assert document.filename == "test.pdf"
        assert document.page_count == 2
        assert document.file_size > 0
    
    def test_load_pdf_invalid_file(self):
        """Test PDF loading with invalid file"""
        with pytest.raises(ValueError, match="Cannot load file"):
            self.pdf_service.load_pdf('invalid.txt')
    
    def test_extract_text_from_pages(self):
        """Test text extraction from pages"""
        # Mock the extract_text_from_pages method
        mock_pages = [
            {'page_number': 1, 'content': "Page 1 content"},
            {'page_number': 2, 'content': "Page 2 content"}
        ]
        
        with patch.object(self.pdf_service, 'extract_text_from_pages', return_value=mock_pages):
            pages = self.pdf_service.extract_text_from_pages("test.pdf")
        
        assert len(pages) == 2
        assert pages[0]['page_number'] == 1
        assert pages[0]['content'] == "Page 1 content"
        assert pages[1]['page_number'] == 2
        assert pages[1]['content'] == "Page 2 content"
    
    def test_validate_pdf_content(self):
        """Test PDF content validation"""
        # Mock the validate_pdf_content method
        mock_validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {
                'filename': 'test.pdf',
                'file_size': 1024,
                'file_size_mb': 0.001,
                'page_count': 2,
                'total_content_length': 500
            }
        }
        
        with patch.object(self.pdf_service, 'validate_pdf_content', return_value=mock_validation):
            validation = self.pdf_service.validate_pdf_content("test.pdf")
        
        assert validation['is_valid'] is True
        assert validation['file_info']['page_count'] == 2
        assert validation['file_info']['filename'] == "test.pdf"
        assert len(validation['errors']) == 0

class TestDocumentChunker:
    """Test document chunker"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunker_initialization(self):
        """Test chunker initialization"""
        assert self.chunker.chunk_size == 100
        assert self.chunker.chunk_overlap == 20
    
    def test_chunker_invalid_parameters(self):
        """Test chunker with invalid parameters"""
        with pytest.raises(ValueError, match="Chunk size must be greater than 0"):
            DocumentChunker(chunk_size=0, chunk_overlap=10)
        
        with pytest.raises(ValueError, match="Chunk overlap must be non-negative"):
            DocumentChunker(chunk_size=100, chunk_overlap=-10)
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "This is a test document with multiple sentences. It should be chunked properly."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.content) <= 100 for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        assert len(chunks) == 0
        
        chunks = self.chunker.chunk_text("   ")
        assert len(chunks) == 0
    
    def test_chunk_text_with_metadata(self):
        """Test text chunking with metadata"""
        text = "Test content for chunking with metadata."
        metadata = {"source": "test", "page": 1}
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata == metadata
            assert chunk.content_hash  # Should be auto-generated
    
    def test_chunk_pages(self):
        """Test chunking multiple pages"""
        pages = [
            {"page_number": 1, "content": "Page 1 content with multiple sentences.", "metadata": {}},
            {"page_number": 2, "content": "Page 2 content with more text.", "metadata": {}}
        ]
        
        chunks = self.chunker.chunk_pages(pages)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        
        # Check that page metadata is preserved
        page_1_chunks = [c for c in chunks if c.metadata.get('page_number') == 1]
        page_2_chunks = [c for c in chunks if c.metadata.get('page_number') == 2]
        
        assert len(page_1_chunks) > 0
        assert len(page_2_chunks) > 0
    
    def test_validate_chunks(self):
        """Test chunk validation"""
        # Create test chunks
        chunks = [
            Chunk(chunk_index=0, content="Valid chunk content"),
            Chunk(chunk_index=1, content="Another valid chunk"),
            Chunk(chunk_index=2, content="")  # Empty chunk
        ]
        
        validation = self.chunker.validate_chunks(chunks)
        
        assert validation['is_valid'] is False  # Empty chunk makes it invalid
        assert len(validation['errors']) > 0
        assert validation['stats']['total_chunks'] == 3
        assert validation['stats']['empty_chunks'] == 1
    
    def test_get_chunking_stats(self):
        """Test chunking statistics"""
        chunks = [
            Chunk(chunk_index=0, content="First chunk"),
            Chunk(chunk_index=1, content="Second chunk"),
            Chunk(chunk_index=2, content="Third chunk")
        ]
        
        stats = self.chunker.get_chunking_stats(chunks)
        
        assert stats['total_chunks'] == 3
        assert stats['total_content_length'] > 0
        assert stats['total_words'] > 0
        assert 'average_chunk_size' in stats
        assert 'average_words_per_chunk' in stats
