from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import json
from typing import Dict, Any
from src.config.settings import Settings

Base = declarative_base()
settings = Settings()


def get_embedding_dimension() -> int:
    """Get the appropriate embedding dimension based on available API keys"""
    if settings.OPENAI_API_KEY:
        return 1536  # OpenAI embedding dimension
    elif settings.GOOGLE_API_KEY:
        return 768   # Google Gemini embedding dimension
    else:
        return 1536  # Default to OpenAI dimension


class DocumentModel(Base):
    """Document model for storing PDF documents"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    total_chunks = Column(Integer, default=0)
    file_size = Column(Integer)  # File size in bytes
    page_count = Column(Integer)  # Number of pages in PDF
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship with chunks
    chunks = relationship(
        "ChunkModel", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', chunks={self.total_chunks})>"


class ChunkModel(Base):
    """Chunk model for storing text chunks with vector embeddings"""

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    embedding = Column(Vector(get_embedding_dimension()), nullable=False)  # Dynamic embedding dimension
    chunk_metadata = Column(Text)  # JSON string for additional info
    page_number = Column(Integer)  # Page number in PDF
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship with document
    document = relationship("DocumentModel", back_populates="chunks")

    # Composite index for efficient document + chunk_index queries
    __table_args__ = (Index("idx_document_chunk_index", "document_id", "chunk_index"),)

    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as dictionary"""
        if self.chunk_metadata:
            try:
                return json.loads(self.chunk_metadata)
            except json.JSONDecodeError:
                return {}
        return {}

    @metadata_dict.setter
    def metadata_dict(self, value: Dict[str, Any]):
        """Set metadata from dictionary"""
        self.chunk_metadata = json.dumps(value) if value else None


class SearchHistoryModel(Base):
    """Search history model for tracking user queries"""

    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    query_embedding = Column(Vector(get_embedding_dimension()), nullable=False)  # Dynamic embedding dimension
    response = Column(Text, nullable=False)
    search_results_count = Column(Integer, default=0)
    processing_time_ms = Column(Integer)  # Processing time in milliseconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<SearchHistory(id={self.id}, query='{self.query[:50]}...')>"
