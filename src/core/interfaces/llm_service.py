from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time


class LLMService(ABC):
    """Abstract interface for Large Language Model services"""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response using LLM with context"""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of the LLM service"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available and configured"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass

    async def generate_response_with_timing(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate response with timing information"""
        start_time = time.time()

        try:
            response = await self.generate_response(
                prompt, context, max_tokens, temperature
            )
            processing_time_ms = int((time.time() - start_time) * 1000)

            return {
                "response": response,
                "processing_time_ms": processing_time_ms,
                "success": True,
                "error": None,
            }

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)

            return {
                "response": None,
                "processing_time_ms": processing_time_ms,
                "success": False,
                "error": str(e),
            }

    def validate_prompt(self, prompt: str) -> bool:
        """Validate if prompt is suitable for LLM processing"""
        if not prompt or not prompt.strip():
            return False

        # Check if prompt is too long (most services have limits)
        if len(prompt) > 32000:  # OpenAI limit for gpt-4
            return False

        return True

    def validate_context(self, context: List[Dict[str, Any]]) -> bool:
        """Validate context data"""
        if context is None:
            return True

        if not isinstance(context, list):
            return False

        for item in context:
            if not isinstance(item, dict):
                return False

            # Check required fields
            if "content" not in item:
                return False

        return True

    def format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Format context data for inclusion in prompt"""
        if not context:
            return ""

        formatted_context = []
        for i, item in enumerate(context, 1):
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            # Format with metadata if available
            if metadata:
                page_info = (
                    f" (Page {metadata.get('page_number', 'N/A')})"
                    if "page_number" in metadata
                    else ""
                )
                formatted_context.append(f"{i}. {content}{page_info}")
            else:
                formatted_context.append(f"{i}. {content}")

        return "\n\n".join(formatted_context)

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def estimate_prompt_tokens(
        self, prompt: str, context: List[Dict[str, Any]] = None
    ) -> int:
        """Estimate total token count for prompt including context"""
        total_tokens = self.estimate_token_count(prompt)

        if context:
            context_text = self.format_context_for_prompt(context)
            total_tokens += self.estimate_token_count(context_text)

        return total_tokens

    def is_prompt_within_limits(
        self, prompt: str, context: List[Dict[str, Any]] = None
    ) -> bool:
        """Check if prompt is within token limits"""
        estimated_tokens = self.estimate_prompt_tokens(prompt, context)

        # Conservative limit (most models support 4k-32k tokens)
        return estimated_tokens < 30000
