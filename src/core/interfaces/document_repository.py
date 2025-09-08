from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk


class DocumentRepository(ABC):
    """Abstract interface for document repository operations"""

    @abstractmethod
    async def save_document(self, document: Document) -> Document:
        """Save document and return saved document with ID"""
        pass

    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Save multiple chunks and return saved chunks with IDs"""
        pass

    @abstractmethod
    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        pass

    @abstractmethod
    async def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash"""
        pass

    @abstractmethod
    async def get_chunks_by_document_id(self, document_id: int) -> List[Chunk]:
        """Get all chunks for a specific document"""
        pass

    @abstractmethod
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks"""
        pass

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update existing document"""
        pass

    @abstractmethod
    async def get_documents_summary(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get summary of all documents"""
        pass

    @abstractmethod
    async def save_search_history(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
        search_results_count: int,
        processing_time_ms: int,
    ) -> int:
        """Save search history and return history ID"""
        pass
