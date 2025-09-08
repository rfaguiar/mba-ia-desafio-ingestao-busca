# Document Chunking System

This document describes the text chunking system used in the RAG application for splitting documents into manageable pieces while preserving context and meaning.

## Overview

The chunking system is responsible for breaking down large text documents into smaller, semantically meaningful chunks that can be:
- Processed by AI models within token limits
- Stored efficiently in the vector database
- Retrieved with proper context during search operations

## Architecture

### Core Components

```python
class DocumentChunker:
    """Main chunking service for text processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
```

### Chunking Strategy

The system uses a **sliding window approach** with configurable parameters:

1. **Chunk Size**: Maximum characters per chunk (default: 1000)
2. **Chunk Overlap**: Overlap between consecutive chunks (default: 200)
3. **Boundary Detection**: Respects sentence and paragraph boundaries

## Chunking Algorithms

### 1. Basic Text Chunking

```python
def chunk_text(self, text: str) -> List[Chunk]:
    """
    Split text into chunks with overlap.
    
    Algorithm:
    1. Start with empty chunk
    2. Add words until chunk_size is reached
    3. Look for sentence boundary (., !, ?)
    4. Create chunk and move to next with overlap
    """
```

**Example:**
```
Input: "This is a long document. It has multiple sentences. Each sentence contains important information."
Chunk 1: "This is a long document. It has multiple sentences."
Chunk 2: "It has multiple sentences. Each sentence contains important information."
```

### 2. Page-Based Chunking

```python
def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Split pages into chunks preserving page metadata.
    
    Args:
        pages: List of page dictionaries with content and metadata
        
    Returns:
        List of chunks with page information preserved
    """
```

**Page Structure:**
```python
{
    "page_number": 1,
    "content": "Page content here...",
    "metadata": {"title": "Chapter 1", "section": "Introduction"}
}
```

### 3. Metadata-Aware Chunking

```python
def chunk_with_metadata(
    self, 
    text: str, 
    metadata: Dict[str, Any]
) -> List[Chunk]:
    """
    Create chunks with attached metadata.
    
    Args:
        text: Text content to chunk
        metadata: Metadata to attach to each chunk
        
    Returns:
        List of chunks with metadata
    """
```

## Configuration

### Default Settings

```python
# From src/config/settings.py
CHUNK_SIZE: int = 1000          # Maximum characters per chunk
CHUNK_OVERLAP: int = 200        # Overlap between chunks
MIN_CHUNK_SIZE: int = 100       # Minimum chunk size
MAX_CHUNK_SIZE: int = 2000      # Maximum chunk size
```

### Custom Configuration

```python
from src.infrastructure.text_processing.chunker import DocumentChunker

# Custom chunking parameters
chunker = DocumentChunker(
    chunk_size=500,      # Smaller chunks for detailed analysis
    chunk_overlap=100    # Less overlap for efficiency
)

# Process text with custom settings
chunks = chunker.chunk_text("Your long text content...")
```

## Chunk Quality Features

### 1. Boundary Detection

The system intelligently breaks chunks at natural boundaries:

- **Sentence Boundaries**: `.`, `!`, `?`
- **Paragraph Boundaries**: Double newlines
- **Section Boundaries**: Headers and titles
- **Semantic Boundaries**: Context switches

### 2. Overlap Management

Overlap ensures context continuity between chunks:

```python
def _calculate_overlap(self, current_chunk: str, next_chunk: str) -> int:
    """
    Calculate optimal overlap between chunks.
    
    Returns:
        Number of characters to overlap
    """
    # Ensure overlap doesn't exceed chunk size
    max_overlap = min(self.chunk_overlap, len(current_chunk) // 2)
    return max_overlap
```

### 3. Content Validation

Each chunk is validated for quality:

```python
def _validate_chunk(self, chunk: Chunk) -> bool:
    """
    Validate chunk quality and content.
    
    Checks:
    - Minimum content length
    - Meaningful content (not just whitespace)
    - Proper metadata
    """
    if len(chunk.content.strip()) < self.min_chunk_size:
        return False
    
    if not chunk.content.strip():
        return False
    
    return True
```

## Performance Considerations

### 1. Memory Management

```python
def chunk_large_document(self, file_path: str) -> Iterator[Chunk]:
    """
    Process large documents without loading entire content into memory.
    
    Uses streaming approach for memory efficiency.
    """
    with open(file_path, 'r') as file:
        buffer = ""
        for line in file:
            buffer += line
            if len(buffer) >= self.chunk_size:
                chunks = self._process_buffer(buffer)
                yield from chunks
                buffer = buffer[-self.chunk_overlap:]
```

### 2. Batch Processing

```python
async def chunk_documents_batch(
    self, 
    documents: List[str]
) -> List[List[Chunk]]:
    """
    Process multiple documents concurrently.
    
    Args:
        documents: List of document contents
        
    Returns:
        List of chunk lists for each document
    """
    tasks = [self.chunk_text(doc) for doc in documents]
    return await asyncio.gather(*tasks)
```

### 3. Caching

```python
@lru_cache(maxsize=1000)
def _get_chunk_signature(self, text: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Generate cache signature for chunking parameters.
    
    Returns:
        Hash string for caching
    """
    return hashlib.md5(f"{text[:100]}_{chunk_size}_{chunk_overlap}".encode()).hexdigest()
```

## Testing and Validation

### 1. Chunk Quality Tests

```python
def test_chunk_quality(self):
    """Test chunk quality and consistency"""
    text = "This is a test document. It has multiple sentences. Each sentence should be properly chunked."
    chunks = self.chunker.chunk_text(text)
    
    # Verify chunk sizes
    for chunk in chunks:
        assert len(chunk.content) <= self.chunk_size
        assert len(chunk.content) >= self.min_chunk_size
    
    # Verify overlap
    for i in range(len(chunks) - 1):
        overlap = self._calculate_overlap(chunks[i].content, chunks[i+1].content)
        assert overlap >= 0
```

### 2. Performance Tests

```python
def test_chunking_performance(self):
    """Test chunking performance with large documents"""
    large_text = "Test content. " * 10000  # 100,000 characters
    
    start_time = time.time()
    chunks = self.chunker.chunk_text(large_text)
    end_time = time.time()
    
    # Should complete within reasonable time
    assert end_time - start_time < 1.0  # Less than 1 second
    assert len(chunks) > 0
```

### 3. Memory Usage Tests

```python
def test_memory_usage(self):
    """Test memory efficiency during chunking"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process large document
    large_text = "Test content. " * 50000  # 500,000 characters
    chunks = self.chunker.chunk_text(large_text)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

## Advanced Features

### 1. Adaptive Chunking

```python
def adaptive_chunk_text(self, text: str, target_tokens: int = 1000) -> List[Chunk]:
    """
    Adapt chunk size based on token count estimation.
    
    Args:
        text: Text to chunk
        target_tokens: Target token count per chunk
        
    Returns:
        List of chunks optimized for token count
    """
    # Estimate tokens per character
    estimated_chars = target_tokens * 4  # Rough estimation
    
    # Adjust chunk size
    adjusted_chunk_size = min(estimated_chars, self.chunk_size)
    
    return self.chunk_text(text, chunk_size=adjusted_chunk_size)
```

### 2. Semantic Chunking

```python
def semantic_chunk_text(self, text: str) -> List[Chunk]:
    """
    Create chunks based on semantic meaning rather than character count.
    
    Uses NLP techniques to identify topic boundaries.
    """
    # Implementation would use NLP libraries like spaCy
    # to identify topic changes and create semantic chunks
    pass
```

### 3. Multi-Language Support

```python
def chunk_multilingual_text(self, text: str, language: str) -> List[Chunk]:
    """
    Handle chunking for different languages.
    
    Args:
        text: Text content
        language: Language code (en, pt, es, etc.)
        
    Returns:
        List of language-appropriate chunks
    """
    # Adjust chunking strategy based on language
    if language in ['zh', 'ja', 'ko']:  # Character-based languages
        return self._chunk_character_based(text)
    else:  # Word-based languages
        return self._chunk_word_based(text)
```

## Monitoring and Metrics

### 1. Chunking Statistics

```python
def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Generate statistics about chunking process.
    
    Returns:
        Dictionary with chunking metrics
    """
    return {
        "total_chunks": len(chunks),
        "average_chunk_size": sum(len(c.content) for c in chunks) / len(chunks),
        "min_chunk_size": min(len(c.content) for c in chunks),
        "max_chunk_size": max(len(c.content) for c in chunks),
        "total_content_length": sum(len(c.content) for c in chunks),
        "compression_ratio": len(chunks) / (sum(len(c.content) for c in chunks) / 1000)
    }
```

### 2. Quality Metrics

```python
def calculate_chunk_quality_score(self, chunk: Chunk) -> float:
    """
    Calculate quality score for a chunk.
    
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 1.0
    
    # Penalize very short chunks
    if len(chunk.content) < self.min_chunk_size:
        score -= 0.3
    
    # Penalize chunks with excessive whitespace
    whitespace_ratio = chunk.content.count(' ') / len(chunk.content)
    if whitespace_ratio > 0.3:
        score -= 0.2
    
    # Bonus for complete sentences
    if chunk.content.strip().endswith(('.', '!', '?')):
        score += 0.1
    
    return max(0.0, min(1.0, score))
```

## Best Practices

### 1. Chunk Size Selection

- **Small chunks (500-800 chars)**: Good for precise search, higher recall
- **Medium chunks (800-1200 chars)**: Balanced approach, good for most use cases
- **Large chunks (1200-2000 chars)**: Better context, higher precision

### 2. Overlap Configuration

- **Low overlap (100-200 chars)**: Efficient storage, good for distinct content
- **Medium overlap (200-400 chars)**: Balanced approach, preserves context
- **High overlap (400+ chars)**: Maximum context preservation, higher storage cost

### 3. Content Preparation

```python
def prepare_text_for_chunking(self, text: str) -> str:
    """
    Prepare text for optimal chunking.
    
    Steps:
    1. Normalize whitespace
    2. Remove excessive line breaks
    3. Ensure proper sentence endings
    4. Clean special characters
    """
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Ensure sentences end properly
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    return text
```
