from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, text, bindparam
from sqlalchemy.orm import selectinload
from pgvector.sqlalchemy import Vector
from typing import List, Optional, Tuple, Dict, Any
import logging
import json
from datetime import datetime

from src.core.interfaces.document_repository import DocumentRepository
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk
from src.infrastructure.database.models.document_model import (
    DocumentModel,
    ChunkModel,
    SearchHistoryModel,
)

logger = logging.getLogger(__name__)


class PostgresDocumentRepository(DocumentRepository):
    """PostgreSQL implementation of document repository with pgVector support"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_document(self, document: Document) -> Document:
        """Save document and return saved document with ID"""
        try:
            # Check if document already exists by content hash
            existing_doc = await self.get_document_by_hash(document.content_hash)
            if existing_doc:
                logger.info(f"Document already exists: {existing_doc.filename}")
                return existing_doc

            # Create database model
            doc_model = DocumentModel(
                filename=document.filename,
                content_hash=document.content_hash,
                total_chunks=document.total_chunks,
                file_size=document.file_size,
                page_count=document.page_count,
            )

            # Save document
            self.session.add(doc_model)
            await self.session.flush()
            await self.session.refresh(doc_model)

            # Update domain entity with ID
            document.id = doc_model.id
            document.created_at = doc_model.created_at
            document.updated_at = doc_model.updated_at

            logger.info(
                f"Document saved successfully: {document.filename} (ID: {document.id})"
            )
            return document

        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            await self.session.rollback()
            raise

    async def save_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Save multiple chunks and return saved chunks with IDs"""
        if not chunks:
            return []

        try:
            chunk_models = []

            for chunk in chunks:
                # Create chunk model
                chunk_model = ChunkModel(
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    embedding=chunk.embedding,
                    chunk_metadata=(
                        json.dumps(chunk.metadata) if chunk.metadata else None
                    ),
                    page_number=chunk.page_number,
                )
                chunk_models.append(chunk_model)

            # Save all chunks
            self.session.add_all(chunk_models)
            await self.session.flush()

            # Update domain entities with IDs
            for i, chunk in enumerate(chunks):
                chunk.id = chunk_models[i].id
                chunk.created_at = chunk_models[i].created_at

            logger.info(f"Successfully saved {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")
            await self.session.rollback()
            raise

    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Get document by ID"""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = await self.session.execute(stmt)
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                return None

            # Convert to domain entity
            document = Document(
                id=doc_model.id,
                filename=doc_model.filename,
                content_hash=doc_model.content_hash,
                total_chunks=doc_model.total_chunks,
                file_size=doc_model.file_size,
                page_count=doc_model.page_count,
                created_at=doc_model.created_at,
                updated_at=doc_model.updated_at,
            )

            return document

        except Exception as e:
            logger.error(f"Error getting document by ID {document_id}: {str(e)}")
            raise

    async def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get document by content hash"""
        try:
            stmt = select(DocumentModel).where(
                DocumentModel.content_hash == content_hash
            )
            result = await self.session.execute(stmt)
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                return None

            # Convert to domain entity
            document = Document(
                id=doc_model.id,
                filename=doc_model.filename,
                content_hash=doc_model.content_hash,
                total_chunks=doc_model.total_chunks,
                file_size=doc_model.file_size,
                page_count=doc_model.page_count,
                created_at=doc_model.created_at,
                updated_at=doc_model.updated_at,
            )

            return document

        except Exception as e:
            logger.error(f"Error getting document by hash {content_hash}: {str(e)}")
            raise

    async def get_chunks_by_document_id(self, document_id: int) -> List[Chunk]:
        """Get all chunks for a specific document"""
        try:
            stmt = (
                select(ChunkModel)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.chunk_index)
            )
            result = await self.session.execute(stmt)
            chunk_models = result.scalars().all()

            chunks = []
            for chunk_model in chunk_models:
                chunk = Chunk(
                    id=chunk_model.id,
                    document_id=chunk_model.document_id,
                    chunk_index=chunk_model.chunk_index,
                    content=chunk_model.content,
                    content_hash=chunk_model.content_hash,
                    embedding=chunk_model.embedding,
                    metadata=(
                        json.loads(chunk_model.chunk_metadata)
                        if chunk_model.chunk_metadata
                        else {}
                    ),
                    page_number=chunk_model.page_number,
                    created_at=chunk_model.created_at,
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
            raise

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity"""
        try:
            # Convert list to string format for pgvector
            query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # Use raw SQL with pgvector operators
            sql = f"""
            SELECT id, document_id, chunk_index, content, content_hash, 
                   embedding, chunk_metadata, page_number, created_at,
                   (1 - (embedding <=> '{query_vector_str}'::vector)) as similarity
            FROM chunks 
            WHERE (1 - (embedding <=> '{query_vector_str}'::vector)) >= {similarity_threshold}
            ORDER BY embedding <=> '{query_vector_str}'::vector ASC
            LIMIT {limit}
            """
            
            result = await self.session.execute(text(sql))
            rows = result.fetchall()

            chunks_with_similarity = []
            for row in rows:
                # Extract data from raw SQL result
                chunk_id, document_id, chunk_index, content, content_hash, embedding, chunk_metadata, page_number, created_at, similarity = row

                # Convert to domain entity
                chunk = Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=content,
                    content_hash=content_hash,
                    embedding=embedding,
                    metadata=(
                        json.loads(chunk_metadata)
                        if chunk_metadata
                        else {}
                    ),
                    page_number=page_number,
                    created_at=created_at,
                )

                chunks_with_similarity.append((chunk, float(similarity)))

            logger.info(f"Found {len(chunks_with_similarity)} similar chunks")
            return chunks_with_similarity

        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise

    async def search_text_chunks(
        self,
        key_terms: List[str],
        limit: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """Search for chunks using text matching"""
        try:
            if not key_terms:
                return []
            
            # Create ILIKE conditions for each key term
            conditions = []
            for term in key_terms:
                conditions.append(ChunkModel.content.ilike(f"%{term}%"))
            
            # Combine conditions with OR
            from sqlalchemy import or_
            combined_condition = or_(*conditions)
            
            # Build query
            stmt = (
                select(ChunkModel)
                .where(combined_condition)
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            chunk_models = result.scalars().all()
            
            chunks_with_similarity = []
            for chunk_model in chunk_models:
                # Calculate text similarity score based on term matches
                content_lower = chunk_model.content.lower()
                matches = sum(1 for term in key_terms if term.lower() in content_lower)
                similarity = matches / len(key_terms) if key_terms else 0.0
                
                # Convert to domain entity
                chunk = Chunk(
                    id=chunk_model.id,
                    document_id=chunk_model.document_id,
                    chunk_index=chunk_model.chunk_index,
                    content=chunk_model.content,
                    content_hash=chunk_model.content_hash,
                    embedding=chunk_model.embedding,
                    metadata=(
                        json.loads(chunk_model.chunk_metadata)
                        if chunk_model.chunk_metadata
                        else {}
                    ),
                    page_number=chunk_model.page_number,
                    created_at=chunk_model.created_at,
                )
                
                chunks_with_similarity.append((chunk, similarity))
            
            # Sort by similarity score (descending)
            chunks_with_similarity.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(chunks_with_similarity)} text matching chunks")
            return chunks_with_similarity

        except Exception as e:
            logger.error(f"Error searching text chunks: {str(e)}")
            raise

    async def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks"""
        try:
            # Delete chunks first (cascade should handle this, but being explicit)
            await self.session.execute(
                delete(ChunkModel).where(ChunkModel.document_id == document_id)
            )

            # Delete document
            await self.session.execute(
                delete(DocumentModel).where(DocumentModel.id == document_id)
            )

            logger.info(f"Document {document_id} and its chunks deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            await self.session.rollback()
            raise

    async def update_document(self, document: Document) -> Document:
        """Update existing document"""
        try:
            if not document.id:
                raise ValueError("Document must have an ID to update")

            stmt = select(DocumentModel).where(DocumentModel.id == document.id)
            result = await self.session.execute(stmt)
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                raise ValueError(f"Document with ID {document.id} not found")

            # Update fields
            doc_model.filename = document.filename
            doc_model.total_chunks = document.total_chunks
            doc_model.file_size = document.file_size
            doc_model.page_count = document.page_count
            doc_model.updated_at = datetime.now()

            await self.session.flush()
            await self.session.refresh(doc_model)

            # Update domain entity
            document.updated_at = doc_model.updated_at

            logger.info(f"Document {document.id} updated successfully")
            return document

        except Exception as e:
            logger.error(f"Error updating document {document.id}: {str(e)}")
            await self.session.rollback()
            raise

    async def get_documents_summary(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get summary of all documents"""
        try:
            stmt = select(DocumentModel).limit(limit)
            result = await self.session.execute(stmt)
            doc_models = result.scalars().all()

            summaries = []
            for doc_model in doc_models:
                summary = {
                    "id": doc_model.id,
                    "filename": doc_model.filename,
                    "total_chunks": doc_model.total_chunks,
                    "file_size": doc_model.file_size,
                    "page_count": doc_model.page_count,
                    "created_at": (
                        doc_model.created_at.isoformat()
                        if doc_model.created_at
                        else None
                    ),
                    "updated_at": (
                        doc_model.updated_at.isoformat()
                        if doc_model.updated_at
                        else None
                    ),
                }
                summaries.append(summary)

            return summaries

        except Exception as e:
            logger.error(f"Error getting documents summary: {str(e)}")
            raise

    async def save_search_history(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
        search_results_count: int,
        processing_time_ms: int,
    ) -> int:
        """Save search history and return history ID"""
        try:
            history_model = SearchHistoryModel(
                query=query,
                query_embedding=query_embedding,
                response=response,
                search_results_count=search_results_count,
                processing_time_ms=processing_time_ms,
            )

            self.session.add(history_model)
            await self.session.flush()
            await self.session.refresh(history_model)

            logger.info(f"Search history saved with ID: {history_model.id}")
            return history_model.id

        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")
            await self.session.rollback()
            raise

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            # Total searches
            total_searches_stmt = select(func.count(SearchHistoryModel.id))
            total_searches_result = await self.session.execute(total_searches_stmt)
            total_searches = total_searches_result.scalar()

            # Average processing time
            avg_time_stmt = select(func.avg(SearchHistoryModel.processing_time_ms))
            avg_time_result = await self.session.execute(avg_time_stmt)
            avg_time = avg_time_result.scalar()

            # Total documents
            total_docs_stmt = select(func.count(DocumentModel.id))
            total_docs_result = await self.session.execute(total_docs_stmt)
            total_docs = total_docs_result.scalar()

            # Total chunks
            total_chunks_stmt = select(func.count(ChunkModel.id))
            total_chunks_result = await self.session.execute(total_chunks_stmt)
            total_chunks = total_chunks_result.scalar()

            stats = {
                "total_searches": total_searches or 0,
                "average_processing_time_ms": round(avg_time, 2) if avg_time else 0,
                "total_documents": total_docs or 0,
                "total_chunks": total_chunks or 0,
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting search statistics: {str(e)}")
            raise
