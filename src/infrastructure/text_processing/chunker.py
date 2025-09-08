from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import hashlib
import logging
from src.core.domain.chunk import Chunk
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Service for splitting documents into text chunks"""

    def __init__(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        )

        # Validate chunk parameters
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"DocumentChunker initialized: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks and return Chunk entities"""
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        try:
            logger.info(f"Chunking text: {len(text)} characters")

            # Split text using LangChain
            text_chunks = self.text_splitter.split_text(text)

            # Convert to Chunk entities
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunk = Chunk(
                    chunk_index=i,
                    content=chunk_text.strip(),
                    metadata=metadata or {},
                    content_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
                )
                chunks.append(chunk)

            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk multiple pages and return Chunk entities"""
        if not pages:
            logger.warning("No pages provided for chunking")
            return []

        try:
            logger.info(f"Chunking {len(pages)} pages")

            all_chunks = []
            for page in pages:
                page_number = page.get("page_number", 0)
                content = page.get("content", "")

                if not content.strip():
                    logger.warning(f"Empty content for page {page_number}")
                    continue

                # Create metadata for this page
                page_metadata = {
                    "page_number": page_number,
                    "source": "pdf",
                    "original_metadata": page.get("metadata", {}),
                }

                # Chunk this page
                page_chunks = self.chunk_text(content, page_metadata)

                # Update chunk indices to be global
                for i, chunk in enumerate(page_chunks):
                    chunk.chunk_index = len(all_chunks) + i

                all_chunks.extend(page_chunks)

            logger.info(
                f"Successfully created {len(all_chunks)} chunks from {len(pages)} pages"
            )
            return all_chunks

        except Exception as e:
            logger.error(f"Error chunking pages: {str(e)}")
            raise

    def chunk_document_content(
        self, content: str, document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Chunk document content with document-level metadata"""
        if not content or not content.strip():
            logger.warning("Empty document content provided")
            return []

        try:
            logger.info(f"Chunking document content: {len(content)} characters")

            # Merge document metadata with chunk metadata
            chunk_metadata = {
                "source": "pdf",
                "document_metadata": document_metadata or {},
            }

            chunks = self.chunk_text(content, chunk_metadata)

            logger.info(
                f"Successfully created {len(chunks)} chunks from document content"
            )
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document content: {str(e)}")
            raise

    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate chunk quality and return validation results"""
        if not chunks:
            return {
                "is_valid": False,
                "errors": ["No chunks provided"],
                "warnings": [],
                "stats": {},
            }

        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_chunks": len(chunks),
                "total_content_length": 0,
                "total_words": 0,
                "chunk_sizes": [],
                "empty_chunks": 0,
                "very_short_chunks": 0,
                "very_long_chunks": 0,
            },
        }

        try:
            for chunk in chunks:
                content_length = chunk.get_content_length()
                word_count = chunk.get_word_count()

                validation_result["stats"]["total_content_length"] += content_length
                validation_result["stats"]["total_words"] += word_count
                validation_result["stats"]["chunk_sizes"].append(content_length)

                # Check for empty chunks
                if content_length == 0:
                    validation_result["stats"]["empty_chunks"] += 1
                    validation_result["errors"].append(
                        f"Empty chunk at index {chunk.chunk_index}"
                    )

                # Check for very short chunks
                if content_length < 10:
                    validation_result["stats"]["very_short_chunks"] += 1
                    validation_result["warnings"].append(
                        f"Very short chunk at index {chunk.chunk_index}: {content_length} chars"
                    )

                # Check for very long chunks
                if content_length > self.chunk_size * 2:
                    validation_result["stats"]["very_long_chunks"] += 1
                    validation_result["warnings"].append(
                        f"Very long chunk at index {chunk.chunk_index}: {content_length} chars"
                    )

            # Calculate averages
            if chunks:
                avg_chunk_size = validation_result["stats"][
                    "total_content_length"
                ] / len(chunks)
                validation_result["stats"]["average_chunk_size"] = round(
                    avg_chunk_size, 2
                )
                validation_result["stats"]["average_words_per_chunk"] = round(
                    validation_result["stats"]["total_words"] / len(chunks), 2
                )

            # Determine if chunks are valid
            if validation_result["stats"]["empty_chunks"] > 0:
                validation_result["is_valid"] = False

            logger.info(f"Chunk validation completed: {len(chunks)} chunks")
            return validation_result

        except Exception as e:
            logger.error(f"Error validating chunks: {str(e)}")
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result

    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunking process"""
        if not chunks:
            return {}

        try:
            stats = {
                "total_chunks": len(chunks),
                "total_content_length": sum(
                    chunk.get_content_length() for chunk in chunks
                ),
                "total_words": sum(chunk.get_word_count() for chunk in chunks),
                "chunk_size_range": {
                    "min": min(chunk.get_content_length() for chunk in chunks),
                    "max": max(chunk.get_content_length() for chunk in chunks),
                },
                "chunk_overlap_ratio": (
                    self.chunk_overlap / self.chunk_size if self.chunk_size > 0 else 0
                ),
                "chunking_parameters": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            }

            # Calculate averages
            if chunks:
                stats["average_chunk_size"] = stats["total_content_length"] / len(
                    chunks
                )
                stats["average_words_per_chunk"] = stats["total_words"] / len(chunks)

            return stats

        except Exception as e:
            logger.error(f"Error calculating chunking stats: {str(e)}")
            return {}
