import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from src.core.domain.chunk import Chunk
from src.infrastructure.services.openai_embedding_service import OpenAIEmbeddingService
from src.infrastructure.services.gemini_embedding_service import GeminiEmbeddingService
from src.infrastructure.services.openai_llm_service import OpenAILLMService
from src.infrastructure.services.gemini_llm_service import GeminiLLMService
from src.infrastructure.database.repositories.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.config.database import get_async_session
from src.config.settings import settings
from src.config.ai_service_selector import get_ai_service
from src.presentation.prompts.search_prompt import SearchPromptTemplate

logger = logging.getLogger(__name__)


class SearchDocumentsUseCase:
    """Use case for searching documents and generating AI responses"""

    def __init__(self):
        # Initialize services based on user preference
        selected_service = get_ai_service()
        
        if selected_service == 'openai':
            self.embedding_service = OpenAIEmbeddingService()
            self.llm_service = OpenAILLMService()
        elif selected_service == 'google':
            self.embedding_service = GeminiEmbeddingService()
            self.llm_service = GeminiLLMService()
        else:
            raise ValueError(
                "No AI service configured. Please set OPENAI_API_KEY or GOOGLE_API_KEY"
            )

    async def execute(self, question: str, limit: int = 10) -> str:
        """Execute document search workflow"""
        start_time = time.time()

        try:
            logger.info(f"Starting document search: {question}")

            # Step 1: Generate embedding for the question
            question_embedding = await self._generate_question_embedding(question)

            # Step 2: Search for similar chunks using hybrid approach
            similar_chunks = await self._hybrid_search_chunks(
                question, question_embedding, limit
            )

            if not similar_chunks:
                return "Não tenho informações necessárias para responder sua pergunta."

            # Step 3: Format context from chunks
            context = self._format_chunks_as_context(similar_chunks)

            # Step 4: Generate AI response
            response = await self._generate_ai_response(question, context)

            # Step 5: Save search history
            processing_time_ms = int((time.time() - start_time) * 1000)
            await self._save_search_history(
                question,
                question_embedding,
                response,
                len(similar_chunks),
                processing_time_ms,
            )

            logger.info(f"Search completed in {processing_time_ms}ms")
            return response

        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise

    async def _generate_question_embedding(self, question: str) -> List[float]:
        """Generate embedding for the search question"""
        try:
            logger.info("Generating question embedding")

            embedding = await self.embedding_service.generate_single_embedding(question)

            logger.info(f"Question embedding generated: {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate question embedding: {str(e)}")
            raise

    async def _hybrid_search_chunks(
        self, question: str, question_embedding: List[float], limit: int
    ) -> List[Tuple[Chunk, float]]:
        """Hybrid search combining semantic and text search"""
        try:
            logger.info(f"Hybrid search for chunks (limit: {limit})")

            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)

                # Extract key terms from question for text search
                key_terms = self._extract_key_terms(question)
                
                # Get semantic results (70% of limit)
                semantic_limit = max(1, int(limit * 0.7))
                semantic_chunks = await repository.search_similar_chunks(
                    question_embedding, limit=semantic_limit, similarity_threshold=0.0
                )
                
                # Get text search results (30% of limit)
                text_limit = max(1, limit - len(semantic_chunks))
                text_chunks = await repository.search_text_chunks(
                    key_terms, limit=text_limit
                )
                
                # Combine and deduplicate results
                combined_chunks = self._combine_search_results(
                    semantic_chunks, text_chunks, limit
                )

                logger.info(f"Found {len(combined_chunks)} chunks (semantic: {len(semantic_chunks)}, text: {len(text_chunks)})")
                return combined_chunks

        except Exception as e:
            logger.error(f"Failed to hybrid search chunks: {str(e)}")
            raise

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question for text search"""
        # Simple extraction - in production, use NLP libraries
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'qual', 'o', 'da', 'de', 'do', 'das', 'dos', 'a', 'as', 'e', 'em', 'na', 'no', 'nas', 'nos', 'para', 'com', 'por', 'que', 'se', 'um', 'uma', 'uns', 'umas'}
        
        # Extract words (3+ characters, not stop words)
        words = re.findall(r'\b\w{3,}\b', question.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms

    def _combine_search_results(
        self, 
        semantic_chunks: List[Tuple[Chunk, float]], 
        text_chunks: List[Tuple[Chunk, float]], 
        limit: int
    ) -> List[Tuple[Chunk, float]]:
        """Combine semantic and text search results, removing duplicates"""
        # Create a dict to store unique chunks by content hash
        unique_chunks = {}
        
        # Add semantic results first (higher priority)
        for chunk, similarity in semantic_chunks:
            unique_chunks[chunk.content_hash] = (chunk, similarity)
        
        # Add text results (lower priority, only if not already present)
        for chunk, similarity in text_chunks:
            if chunk.content_hash not in unique_chunks:
                unique_chunks[chunk.content_hash] = (chunk, similarity)
        
        # Convert back to list and sort by similarity (descending)
        combined = list(unique_chunks.values())
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:limit]

    async def _search_similar_chunks(
        self, question_embedding: List[float], limit: int
    ) -> List[Tuple[Chunk, float]]:
        """Search for chunks similar to the question (legacy method)"""
        try:
            logger.info(f"Searching for similar chunks (limit: {limit})")

            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)

                # Search for similar chunks
                similar_chunks = await repository.search_similar_chunks(
                    question_embedding, limit=limit, similarity_threshold=0.0
                )

                logger.info(f"Found {len(similar_chunks)} similar chunks")
                return similar_chunks

        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            raise

    def _format_chunks_as_context(
        self, chunks_with_similarity: List[Tuple[Chunk, float]]
    ) -> List[Dict[str, Any]]:
        """Format chunks for context inclusion"""
        try:
            context = []

            for chunk, similarity in chunks_with_similarity:
                context_item = {
                    "content": chunk.content,
                    "metadata": {
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "similarity_score": round(similarity, 4),
                        "document_id": chunk.document_id,
                    },
                }
                context.append(context_item)

            logger.info(f"Formatted {len(context)} chunks for context")
            return context

        except Exception as e:
            logger.error(f"Failed to format chunks as context: {str(e)}")
            raise

    async def _generate_ai_response(
        self, question: str, context: List[Dict[str, Any]]
    ) -> str:
        """Generate AI response using LLM service"""
        try:
            logger.info("Generating AI response")

            # Create prompt using template
            prompt = SearchPromptTemplate.format_prompt(question, context)

            # Check if prompt is within limits
            if not self.llm_service.is_prompt_within_limits(prompt, context):
                # Truncate context if needed
                context = SearchPromptTemplate.truncate_context_if_needed(
                    context, question
                )
                prompt = SearchPromptTemplate.format_prompt(question, context)
                logger.warning("Context truncated due to prompt length limits")

            # Generate response
            response = await self.llm_service.generate_response(prompt, context)

            logger.info("AI response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Failed to generate AI response: {str(e)}")
            raise

    async def _save_search_history(
        self,
        question: str,
        question_embedding: List[float],
        response: str,
        search_results_count: int,
        processing_time_ms: int,
    ):
        """Save search history to database"""
        try:
            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)

                await repository.save_search_history(
                    question,
                    question_embedding,
                    response,
                    search_results_count,
                    processing_time_ms,
                )

                logger.info("Search history saved")

        except Exception as e:
            logger.warning(f"Failed to save search history: {str(e)}")
            # Don't fail the search if history saving fails

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            async for session in get_async_session():
                repository = PostgresDocumentRepository(session)

                stats = await repository.get_search_statistics()
                return stats

        except Exception as e:
            logger.error(f"Failed to get search statistics: {str(e)}")
            return {}

    async def search_with_metadata(
        self, question: str, limit: int = 10, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Search with additional metadata in response"""
        try:
            start_time = time.time()

            # Execute basic search
            question_embedding = await self._generate_question_embedding(question)
            similar_chunks = await self._search_similar_chunks(
                question_embedding, limit
            )

            if not similar_chunks:
                return {
                    "response": "Não tenho informações necessárias para responder sua pergunta.",
                    "context_chunks": [],
                    "metadata": {
                        "search_time_ms": int((time.time() - start_time) * 1000),
                        "chunks_found": 0,
                        "embedding_service": self.embedding_service.get_service_name(),
                        "llm_service": self.llm_service.get_service_name(),
                    },
                }

            # Format context
            context = self._format_chunks_as_context(similar_chunks)

            # Generate response
            response = await self._generate_ai_response(question, context)

            # Prepare result
            result = {
                "response": response,
                "context_chunks": context if include_metadata else [],
                "metadata": {
                    "search_time_ms": int((time.time() - start_time) * 1000),
                    "chunks_found": len(similar_chunks),
                    "embedding_service": self.embedding_service.get_service_name(),
                    "llm_service": self.llm_service.get_service_name(),
                    "top_similarity_score": max(
                        similarity for _, similarity in similar_chunks
                    ),
                    "average_similarity_score": sum(
                        similarity for _, similarity in similar_chunks
                    )
                    / len(similar_chunks),
                },
            }

            # Save search history
            processing_time_ms = int((time.time() - start_time) * 1000)
            await self._save_search_history(
                question,
                question_embedding,
                response,
                len(similar_chunks),
                processing_time_ms,
            )

            return result

        except Exception as e:
            logger.error(f"Search with metadata failed: {str(e)}")
            raise

    async def validate_search_capabilities(self) -> Dict[str, Any]:
        """Validate that search capabilities are working"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "services": {},
            }

            # Test embedding service
            try:
                test_embedding = await self.embedding_service.generate_single_embedding(
                    "test"
                )
                validation_result["services"]["embedding"] = {
                    "status": "working",
                    "dimension": len(test_embedding),
                    "service_name": self.embedding_service.get_service_name(),
                }
            except Exception as e:
                validation_result["services"]["embedding"] = {
                    "status": "error",
                    "error": str(e),
                }
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Embedding service error: {str(e)}")

            # Test LLM service
            try:
                test_response = await self.llm_service.generate_response("test")
                validation_result["services"]["llm"] = {
                    "status": "working",
                    "service_name": self.llm_service.get_service_name(),
                }
            except Exception as e:
                validation_result["services"]["llm"] = {
                    "status": "error",
                    "error": str(e),
                }
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"LLM service error: {str(e)}")

            # Test database connection
            try:
                async for session in get_async_session():
                    repository = PostgresDocumentRepository(session)
                    stats = await repository.get_search_statistics()
                    validation_result["services"]["database"] = {
                        "status": "working",
                        "total_documents": stats.get("total_documents", 0),
                        "total_chunks": stats.get("total_chunks", 0),
                    }
                    break
            except Exception as e:
                validation_result["services"]["database"] = {
                    "status": "error",
                    "error": str(e),
                }
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Database error: {str(e)}")

            return validation_result

        except Exception as e:
            logger.error(f"Search capabilities validation failed: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "services": {},
            }
