# RAG System API Documentation

This document describes the internal API structure and interfaces for the RAG (Retrieval-Augmented Generation) system.

## Table of Contents

- [Core Interfaces](#core-interfaces)
- [Use Cases](#use-cases)
- [Infrastructure Services](#infrastructure-services)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

## Core Interfaces

### Document Repository Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk

class DocumentRepository(ABC):
    """Abstract interface for document storage operations"""
    
    @abstractmethod
    async def save_document(self, document: Document) -> Document:
        """Save a document to storage"""
        pass
    
    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Save document chunks to storage"""
        pass
    
    @abstractmethod
    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Retrieve document by ID"""
        pass
    
    @abstractmethod
    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity"""
        pass
```

### Embedding Service Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class EmbeddingService(ABC):
    """Abstract interface for embedding generation"""
    
    @abstractmethod
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if service is available"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model information"""
        pass
```

### LLM Service Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMService(ABC):
    """Abstract interface for language model operations"""
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate response using LLM"""
        pass
    
    @abstractmethod
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt length and content"""
        pass
    
    @abstractmethod
    def validate_context(self, context: List[Dict[str, Any]]) -> bool:
        """Validate context for LLM input"""
        pass
    
    @abstractmethod
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        pass
```

## Use Cases

### PDF Ingestion Use Case

```python
class IngestPDFUseCase:
    """Business logic for PDF ingestion workflow"""
    
    def __init__(self):
        self.pdf_service = PDFLoaderService()
        self.chunker = DocumentChunker()
        self.embedding_service = self._get_embedding_service()
        self.repository = PostgresDocumentRepository()
    
    async def execute(self, pdf_path: str) -> Document:
        """
        Execute PDF ingestion workflow
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document: Ingested document with chunks
            
        Raises:
            ValueError: If PDF is invalid or processing fails
            FileNotFoundError: If PDF file doesn't exist
        """
        # Implementation details...
```

**Workflow Steps:**
1. **Validate PDF**: Check file format and content
2. **Load Content**: Extract text from PDF pages
3. **Chunk Text**: Split content into manageable chunks
4. **Generate Embeddings**: Create vector representations
5. **Store Data**: Save document and chunks to database

### Document Search Use Case

```python
class SearchDocumentsUseCase:
    """Business logic for document search and RAG response generation"""
    
    def __init__(self):
        self.embedding_service = self._get_embedding_service()
        self.llm_service = self._get_llm_service()
        self.repository = PostgresDocumentRepository()
    
    async def execute(self, question: str, limit: int = 5) -> str:
        """
        Execute document search workflow
        
        Args:
            question: User's search question
            limit: Maximum number of chunks to retrieve
            
        Returns:
            str: Generated response based on retrieved context
        """
        # Implementation details...
```

**Workflow Steps:**
1. **Generate Query Embedding**: Convert question to vector
2. **Search Similar Chunks**: Find relevant document chunks
3. **Format Context**: Prepare context for LLM
4. **Generate Response**: Use LLM to create answer
5. **Return Result**: Provide contextual response

## Infrastructure Services

### PDF Loader Service

```python
class PDFLoaderService:
    """Service for loading and processing PDF documents"""
    
    def load_pdf(self, pdf_path: str) -> Document:
        """Load PDF and create document entity"""
        pass
    
    def extract_text_from_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content from PDF pages"""
        pass
    
    def validate_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Validate PDF content and return metadata"""
        pass
```

### Document Chunker Service

```python
class DocumentChunker:
    """Service for splitting documents into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into chunks with overlap"""
        pass
    
    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Split pages into chunks preserving metadata"""
        pass
```

### Database Connection Service

```python
class DatabaseConnection:
    """Service for database connection management"""
    
    @staticmethod
    async def get_async_session():
        """Get async database session"""
        pass
    
    @staticmethod
    async def init_database():
        """Initialize database schema"""
        pass
    
    @staticmethod
    async def close_database():
        """Close database connections"""
        pass
```

## Data Models

### Document Entity

```python
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
    chunks: List[Chunk] = field(default_factory=list)
```

**Properties:**
- **id**: Unique database identifier
- **filename**: Original PDF filename
- **content_hash**: SHA256 hash of document content
- **total_chunks**: Number of text chunks
- **file_size**: File size in bytes
- **page_count**: Number of PDF pages
- **chunks**: List of associated text chunks

### Chunk Entity

```python
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
```

**Properties:**
- **id**: Unique database identifier
- **document_id**: Reference to parent document
- **chunk_index**: Position in document sequence
- **content**: Text content of chunk
- **embedding**: Vector representation (1536 dimensions)
- **metadata**: Additional chunk information
- **page_number**: Source PDF page

## Error Handling

### Exception Hierarchy

```python
class RAGSystemError(Exception):
    """Base exception for RAG system"""
    pass

class PDFProcessingError(RAGSystemError):
    """Error during PDF processing"""
    pass

class EmbeddingGenerationError(RAGSystemError):
    """Error during embedding generation"""
    pass

class DatabaseError(RAGSystemError):
    """Error during database operations"""
    pass

class ValidationError(RAGSystemError):
    """Error during data validation"""
    pass
```

### Error Response Format

```python
{
    "error": {
        "type": "PDFProcessingError",
        "message": "Failed to load PDF file",
        "details": "File appears to be corrupted",
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "req_12345"
    }
}
```

## Configuration

### Settings Class

```python
class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/rag_db"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # AI Service Configuration
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Search Configuration
    SIMILARITY_THRESHOLD: float = 0.0
    MAX_SEARCH_RESULTS: int = 10
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT: int = 30
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - |
| `CHUNK_SIZE` | Maximum characters per chunk | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `SIMILARITY_THRESHOLD` | Minimum similarity for search | 0.0 |

## Search and Retrieval

### Vector Similarity Search

```python
async def search_similar_chunks(
    self,
    query_embedding: List[float],
    limit: int = 10,
    similarity_threshold: float = 0.0
) -> List[Tuple[Chunk, float]]:
    """
    Search for similar chunks using cosine similarity
    
    Args:
        query_embedding: Query vector (1536 dimensions)
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of (chunk, similarity_score) tuples
    """
```

**Search Algorithm:**
1. **Cosine Similarity**: Calculate similarity between query and chunk embeddings
2. **Threshold Filtering**: Filter results above similarity threshold
3. **Ranking**: Sort by similarity score (descending)
4. **Limiting**: Return top N results

### Context Formatting

```python
def format_context_for_prompt(
    self, 
    chunks: List[Tuple[Chunk, float]]
) -> List[Dict[str, Any]]:
    """
    Format retrieved chunks for LLM prompt
    
    Args:
        chunks: List of (chunk, similarity) tuples
        
    Returns:
        Formatted context for LLM input
    """
```

**Context Structure:**
```python
[
    {
        "content": "Chunk text content...",
        "metadata": {
            "page_number": 1,
            "chunk_index": 0,
            "similarity": 0.85
        }
    }
]
```

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Process multiple chunks simultaneously
2. **Connection Pooling**: Reuse database connections
3. **Caching**: Cache frequently accessed embeddings
4. **Indexing**: Use database indexes for vector search
5. **Async Operations**: Non-blocking I/O operations

### Monitoring Metrics

- **Response Time**: Time to generate search results
- **Throughput**: Requests processed per second
- **Memory Usage**: RAM consumption during operations
- **Database Performance**: Query execution times
- **Error Rates**: Percentage of failed requests

