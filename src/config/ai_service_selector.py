"""
AI Service Selector

Module to handle AI service selection based on user preference and availability.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

ServiceType = Literal['openai', 'google']


class AIServiceSelector:
    """Handles AI service selection logic."""
    
    @staticmethod
    def get_preferred_service() -> Optional[ServiceType]:
        """Get the user's preferred AI service from .preferred_service file."""
        preferred_file = Path('.preferred_service')
        
        if not preferred_file.exists():
            return None
        
        try:
            with open(preferred_file, 'r') as f:
                service = f.read().strip().lower()
            
            if service in ['openai', 'google']:
                return service
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def get_available_services() -> list[ServiceType]:
        """Get list of available AI services based on API keys."""
        services = []
        
        if os.getenv('OPENAI_API_KEY'):
            services.append('openai')
        
        if os.getenv('GOOGLE_API_KEY'):
            services.append('google')
        
        return services
    
    @staticmethod
    def select_service() -> Optional[ServiceType]:
        """
        Select the AI service to use based on preference and availability.
        
        Priority:
        1. User's preferred service (if available)
        2. First available service
        3. None if no services available
        """
        available = AIServiceSelector.get_available_services()
        
        if not available:
            return None
        
        preferred = AIServiceSelector.get_preferred_service()
        
        # If preferred service is available, use it
        if preferred and preferred in available:
            return preferred
        
        # Otherwise, use first available service
        return available[0]
    
    @staticmethod
    def get_service_info() -> dict:
        """Get information about current service selection."""
        available = AIServiceSelector.get_available_services()
        preferred = AIServiceSelector.get_preferred_service()
        selected = AIServiceSelector.select_service()
        
        return {
            'available_services': available,
            'preferred_service': preferred,
            'selected_service': selected,
            'openai_configured': 'openai' in available,
            'google_configured': 'google' in available
        }


def get_ai_service() -> Optional[ServiceType]:
    """Convenience function to get the selected AI service."""
    return AIServiceSelector.select_service()


def get_service_info() -> dict:
    """Convenience function to get service information."""
    return AIServiceSelector.get_service_info()
