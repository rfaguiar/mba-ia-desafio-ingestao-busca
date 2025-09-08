import asyncio
import logging
from typing import Optional
from pathlib import Path

from src.core.domain.document import Document
from src.core.domain.chunk import Chunk
from src.infrastructure.services.pdf_loader_service import PDFLoaderService
from src.infrastructure.text_processing.chunker import DocumentChunker
from src.infrastructure.services.openai_embedding_service import OpenAIEmbeddingService
from src.infrastructure.services.gemini_embedding_service import GeminiEmbeddingService
from src.infrastructure.database.repositories.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.config.database import get_async_session
from src.config.settings import settings
from src.config.ai_service_selector import get_ai_service

logger = logging.getLogger(__name__)


class IngestPDFUseCase:
    """
    Use case for ingesting PDF documents into the vector database.

    This class orchestrates the complete PDF ingestion workflow including:
    - PDF loading and validation
    - Text extraction and chunking
    - Embedding generation
    - Database storage

    Attributes:
        pdf_service: Service for loading and processing PDF files
        chunker: Service for splitting text into chunks
        embedding_service: Service for generating text embeddings
    """

    def __init__(self) -> None:
        """
        Initialize the PDF ingestion use case.

        Raises:
            ValueError: If no AI service (OpenAI or Google) is configured
        """
        self.pdf_service = PDFLoaderService()
        self.chunker = DocumentChunker()

        # Initialize embedding service based on user preference
        selected_service = get_ai_service()
        
        if selected_service == 'openai':
            self.embedding_service = OpenAIEmbeddingService()
        elif selected_service == 'google':
            self.embedding_service = GeminiEmbeddingService()
        else:
            raise ValueError(
                "No AI service configured. Please set OPENAI_API_KEY or GOOGLE_API_KEY"
            )

    async def execute(
        self,
        pdf_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        force: bool = False,
    ) -> Document:
        """
        Execute the complete PDF ingestion workflow.

        This method performs the following steps:
        1. Load and validate the PDF file
        2. Check for existing documents (unless force=True)
        3. Extract text content from PDF pages
        4. Split text into chunks with specified parameters
        5. Generate embeddings for each chunk
        6. Store document and chunks in the database

        Args:
            pdf_path: Path to the PDF file to ingest
            chunk_size: Custom chunk size (overrides default settings)
            chunk_overlap: Custom chunk overlap (overrides default settings)
            force: If True, re-ingest even if document exists

        Returns:
            Document: The ingested document with metadata and chunks

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF is invalid or processing fails
            Exception: For other processing or database errors

        Example:
            >>> use_case = IngestPDFUseCase()
            >>> document = await use_case.execute("document.pdf", chunk_size=500)
            >>> print(f"Ingested {document.total_chunks} chunks")
        """
        try:
            logger.info(f"Starting PDF ingestion: {pdf_path}")

            # Step 1: Load and validate PDF
            document = await self._load_pdf(pdf_path)

            # Step 2: Check if document already exists
            if not force:
                existing_doc = await self._check_existing_document(
                    document.content_hash
                )
                if existing_doc:
                    logger.info(f"Document already exists: {existing_doc.filename}")
                    return existing_doc

            # Step 3: Extract text from PDF
            pages = self.pdf_service.extract_text_from_pages(pdf_path)

            # Step 4: Chunk the text
            chunks = self.chunker.chunk_pages(pages)

            # Step 5: Generate embeddings for chunks
            await self._generate_embeddings_for_chunks(chunks)

            # Step 6: Save document and chunks to database
            saved_document = await self._save_to_database(document, chunks)

            logger.info(f"Successfully ingested PDF: {saved_document.filename}")
            return saved_document

        except Exception as e:
            logger.error(f"PDF ingestion failed: {str(e)}")
            raise

    async def _load_pdf(self, pdf_path: str) -> Document:
        """
        Load PDF file and create document entity.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document: Document entity with metadata

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If the PDF file is invalid or corrupted
        """
        try:
            # Validate file exists
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Load PDF using service
            document = self.pdf_service.load_pdf(pdf_path)

            logger.info(
                f"PDF loaded: {document.filename} ({document.page_count} pages)"
            )
            return document

        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            raise

    async def _check_existing_document(self, content_hash: str) -> Optional[Document]:
        """Check if document already exists in database"""
        try:
            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)
                existing_doc = await repository.get_document_by_hash(content_hash)
                return existing_doc

        except Exception as e:
            logger.warning(f"Failed to check existing document: {str(e)}")
            return None

    async def _generate_embeddings_for_chunks(self, chunks: list[Chunk]):
        """Generate embeddings for all chunks"""
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings in batches
            embeddings = await self.embedding_service.generate_embeddings_batched(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.set_embedding(embedding)

            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    async def _save_to_database(
        self, document: Document, chunks: list[Chunk]
    ) -> Document:
        """Save document and chunks to database"""
        try:
            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)

                # Save document first
                saved_document = await repository.save_document(document)

                # Set document ID for chunks
                for chunk in chunks:
                    chunk.document_id = saved_document.id

                # Save chunks
                saved_chunks = await repository.save_chunks(chunks)

                # Update document with chunk count
                saved_document.total_chunks = len(saved_chunks)
                saved_document.chunks = saved_chunks

                # Update document in database
                updated_document = await repository.update_document(saved_document)

                # Explicitly commit the transaction
                await session.commit()

                logger.info(
                    f"Saved to database: {updated_document.filename} with {len(saved_chunks)} chunks"
                )
                return updated_document

        except Exception as e:
            logger.error(f"Failed to save to database: {str(e)}")
            raise

    async def validate_ingestion(self, document: Document, chunks: list[Chunk]) -> dict:
        """Validate ingestion results"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "stats": {},
            }

            # Validate document
            if not document.filename:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Document filename is missing")

            if not document.content_hash:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Document content hash is missing")

            # Validate chunks
            if not chunks:
                validation_result["is_valid"] = False
                validation_result["errors"].append("No chunks generated")
            else:
                # Validate chunk quality
                chunk_validation = self.chunker.validate_chunks(chunks)
                validation_result["errors"].extend(chunk_validation["errors"])
                validation_result["warnings"].extend(chunk_validation["warnings"])

                if not chunk_validation["is_valid"]:
                    validation_result["is_valid"] = False

                # Add chunk statistics
                validation_result["stats"] = chunk_validation["stats"]

            # Check embedding dimensions
            if chunks and chunks[0].has_embedding():
                expected_dimension = self.embedding_service.get_embedding_dimension()
                actual_dimension = len(chunks[0].get_embedding())

                if actual_dimension != expected_dimension:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(
                        f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dimension}"
                    )

            return validation_result

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "stats": {},
            }

    def get_ingestion_stats(self, document: Document, chunks: list[Chunk]) -> dict:
        """Get statistics about the ingestion process"""
        try:
            stats = {
                "document": {
                    "filename": document.filename,
                    "file_size": document.file_size,
                    "page_count": document.page_count,
                    "total_chunks": len(chunks),
                },
                "chunking": self.chunker.get_chunking_stats(chunks),
                "embedding_service": self.embedding_service.get_service_name(),
                "embedding_dimension": self.embedding_service.get_embedding_dimension(),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {str(e)}")
            return {}
