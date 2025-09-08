from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import hashlib
import json


@dataclass
class Chunk:
    """Text chunk domain entity"""

    id: Optional[int] = None
    document_id: Optional[int] = None
    chunk_index: int = 0
    content: str = ""
    content_hash: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_number: Optional[int] = None
    created_at: Optional[datetime] = None
    document: Optional[Any] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.content_hash and self.content:
            self.content_hash = self._generate_content_hash()

        if not self.created_at:
            self.created_at = datetime.now()

    def _generate_content_hash(self) -> str:
        """Generate content hash from chunk content"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair"""
        self.metadata[key] = value

    def remove_metadata(self, key: str):
        """Remove metadata key"""
        if key in self.metadata:
            del self.metadata[key]

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with default"""
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists"""
        return key in self.metadata

    def set_embedding(self, embedding: List[float]):
        """Set chunk embedding vector"""
        self.embedding = embedding

    def get_embedding(self) -> Optional[List[float]]:
        """Get chunk embedding vector"""
        return self.embedding

    def has_embedding(self) -> bool:
        """Check if chunk has embedding"""
        return self.embedding is not None

    def get_content_length(self) -> int:
        """Get content length in characters"""
        return len(self.content)

    def get_word_count(self) -> int:
        """Get word count in content"""
        return len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_hash": self.content_hash,
            "embedding_length": len(self.embedding) if self.embedding else 0,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "content_length": self.get_content_length(),
            "word_count": self.get_word_count(),
        }

    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Chunk):
            return False
        return self.content_hash == other.content_hash

    def __hash__(self):
        """Hash based on content hash"""
        return hash(self.content_hash)
