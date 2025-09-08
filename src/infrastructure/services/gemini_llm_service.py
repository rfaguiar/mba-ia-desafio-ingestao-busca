from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.interfaces.llm_service import LLMService
from src.config.settings import settings
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GeminiLLMService(LLMService):
    """Google Gemini LLM service implementation"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model = model or settings.GOOGLE_LLM_MODEL

        if not self.api_key:
            raise ValueError("Google API key is required")

        # Initialize Gemini client
        self.client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=0.1,  # Low temperature for consistent responses
            max_output_tokens=1000,  # Reasonable limit for responses
        )

        logger.info(f"Gemini LLM Service initialized with model: {self.model}")

    async def generate_response(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response using Gemini LLM with context"""
        try:
            # Validate inputs
            if not self.validate_prompt(prompt):
                raise ValueError("Invalid prompt provided")

            if not self.validate_context(context):
                raise ValueError("Invalid context provided")

            # Check token limits
            if not self.is_prompt_within_limits(prompt, context):
                raise ValueError("Prompt exceeds token limits")

            # Format context if provided
            if context:
                formatted_context = self.format_context_for_prompt(context)
                full_prompt = f"{prompt}\n\nContext:\n{formatted_context}"
            else:
                full_prompt = prompt

            logger.info(f"Generating response using Gemini model: {self.model}")

            # Generate response
            response = await self.client.ainvoke(full_prompt)

            # Extract response content
            if hasattr(response, "content"):
                response_text = response.content
            elif hasattr(response, "text"):
                response_text = response.text
            else:
                response_text = str(response)

            logger.info("Successfully generated response using Gemini")
            return response_text.strip()

        except Exception as e:
            logger.error(f"Failed to generate response using Gemini: {str(e)}")
            raise Exception(f"Gemini response generation failed: {str(e)}")

    def get_service_name(self) -> str:
        """Get the name of the LLM service"""
        return "Google Gemini"

    def is_available(self) -> bool:
        """Check if the service is available and configured"""
        return bool(self.api_key) and self.api_key != "your_google_api_key_here"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "service": "Google Gemini",
            "model": self.model,
            "api_key_configured": bool(self.api_key),
            "service_available": self.is_available(),
            "max_tokens": 1000,
            "temperature": 0.1,
        }

    async def test_connection(self) -> bool:
        """Test the connection to Gemini service"""
        try:
            test_prompt = (
                "Hello, this is a test. Please respond with 'Test successful'."
            )
            response = await self.generate_response(test_prompt)

            if response and "test successful" in response.lower():
                logger.info("Gemini connection test successful")
                return True
            else:
                logger.error("Gemini connection test failed: unexpected response")
                return False

        except Exception as e:
            logger.error(f"Gemini connection test failed: {str(e)}")
            return False

    def get_token_limits(self) -> Dict[str, int]:
        """Get token limits for the current model"""
        # Gemini model token limits
        limits = {
            "gemini-2.0-flash-exp": 1048576,  # 1M tokens
            "gemini-2.0-flash": 1048576,  # 1M tokens
            "gemini-1.5-flash": 1048576,  # 1M tokens
            "gemini-1.5-pro": 1048576,  # 1M tokens
            "gemini-2.5-flash-lite": 1048576,  # 1M tokens
        }

        return {
            "max_tokens": limits.get(self.model, 1048576),
            "max_prompt_tokens": limits.get(self.model, 1048576)
            - 1000,  # Reserve for response
            "max_response_tokens": 1000,
        }

    def optimize_prompt_for_model(
        self, prompt: str, context: List[Dict[str, Any]] = None
    ) -> str:
        """Optimize prompt for the specific model"""
        if not context:
            return prompt

        # Estimate tokens
        estimated_tokens = self.estimate_prompt_tokens(prompt, context)
        token_limits = self.get_token_limits()

        # If within limits, return as is
        if estimated_tokens <= token_limits["max_prompt_tokens"]:
            return prompt

        # If over limit, truncate context
        logger.warning(
            f"Prompt exceeds token limit ({estimated_tokens} > {token_limits['max_prompt_tokens']}), truncating context"
        )

        # Calculate how many context items we can include
        prompt_tokens = self.estimate_token_count(prompt)
        available_tokens = token_limits["max_prompt_tokens"] - prompt_tokens

        # Truncate context to fit within limits
        truncated_context = []
        current_tokens = 0

        for item in context:
            item_tokens = self.estimate_token_count(item.get("content", ""))
            if current_tokens + item_tokens <= available_tokens:
                truncated_context.append(item)
                current_tokens += item_tokens
            else:
                break

        # Format truncated context
        if truncated_context:
            formatted_context = self.format_context_for_prompt(truncated_context)
            return f"{prompt}\n\nContext (truncated):\n{formatted_context}"
        else:
            return prompt
