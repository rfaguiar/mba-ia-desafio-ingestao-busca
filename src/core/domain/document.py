from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import hashlib
import os


@dataclass
class Document:
    """Document domain entity"""

    id: Optional[int] = None
    filename: str = ""
    content_hash: str = ""
    total_chunks: int = 0
    file_size: int = 0
    page_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    chunks: List[Any] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.content_hash and self.filename:
            self.content_hash = self._generate_content_hash()

        if not self.created_at:
            self.created_at = datetime.now()

        if not self.updated_at:
            self.updated_at = datetime.now()

    def _generate_content_hash(self) -> str:
        """Generate content hash from filename"""
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        return ""

    def add_chunk(self, chunk: Any):
        """Add chunk to document"""
        chunk.document_id = self.id
        chunk.document = self
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)

    def remove_chunk(self, chunk: Any):
        """Remove chunk from document"""
        if chunk in self.chunks:
            self.chunks.remove(chunk)
            self.total_chunks = len(self.chunks)

    def get_chunk_by_index(self, index: int) -> Optional[Any]:
        """Get chunk by index"""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None

    def get_chunks_by_page(self, page_number: int) -> List[Any]:
        """Get all chunks for a specific page"""
        return [chunk for chunk in self.chunks if chunk.page_number == page_number]

    def update_metadata(self, **kwargs):
        """Update document metadata"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "id": self.id,
            "filename": self.filename,
            "content_hash": self.content_hash,
            "total_chunks": self.total_chunks,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "chunks_count": len(self.chunks),
        }

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', chunks={self.total_chunks})>"
