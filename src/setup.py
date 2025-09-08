#!/usr/bin/env python3
"""
LLM Service Setup Script

Simple script to configure which LLM service to use for subsequent commands.
This script only sets the preferred service, it doesn't modify .env configuration.
Usage: 
    python src/setup.py openai
    python src/setup.py google
    python src/setup.py auto
    python src/setup.py status
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load existing .env
load_dotenv()


def get_available_services():
    """Get list of available AI services based on .env configuration."""
    services = []
    
    if os.getenv('OPENAI_API_KEY'):
        services.append('openai')
    
    if os.getenv('GOOGLE_API_KEY'):
        services.append('google')
    
    return services


def show_current_config():
    """Show current configuration status."""
    print("=" * 50)
    print("Current AI Service Configuration")
    print("=" * 50)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    print(f"OpenAI API Key: {'✓ Configured' if openai_key else '✗ Not configured'}")
    print(f"Google API Key: {'✓ Configured' if google_key else '✗ Not configured'}")
    
    # Check if there's a preferred service file
    preferred_file = Path('.preferred_service')
    if preferred_file.exists():
        with open(preferred_file, 'r') as f:
            preferred = f.read().strip()
        print(f"Preferred Service: {preferred.upper()}")
    else:
        print("Preferred Service: Not set (will auto-detect)")
    
    print()


def set_preferred_service(service):
    """Set the preferred AI service."""
    preferred_file = Path('.preferred_service')
    
    with open(preferred_file, 'w') as f:
        f.write(service)
    
    print(f"Preferred service set to: {service.upper()}")
    print(f"Configuration saved to: {preferred_file.absolute()}")





def setup_service(service_name: str):
    """Set up the preferred AI service."""
    available = get_available_services()
    
    if not available:
        print("No AI services configured!")
        print("Please configure at least one API key in your .env file:")
        print("- OPENAI_API_KEY for OpenAI")
        print("- GOOGLE_API_KEY for Google Gemini")
        sys.exit(1)
    
    service_name = service_name.lower()
    
    if service_name == 'auto':
        # Remove preferred service file to enable auto-detection
        preferred_file = Path('.preferred_service')
        if preferred_file.exists():
            preferred_file.unlink()
        print("Auto-detection enabled (will use first available service)")
        return
    
    if service_name not in available:
        print(f"Service '{service_name}' is not available!")
        print(f"Available services: {', '.join(available)}")
        print("Make sure the corresponding API key is configured in your .env file")
        sys.exit(1)
    
    set_preferred_service(service_name)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Configure preferred AI service for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/setup.py openai     # Set OpenAI as preferred service
  python src/setup.py google     # Set Google Gemini as preferred service  
  python src/setup.py auto       # Enable auto-detection
  python src/setup.py status     # Show current configuration
        """
    )
    
    parser.add_argument(
        'service',
        nargs='?',
        choices=['openai', 'google', 'auto', 'status'],
        help='AI service to set as preferred (openai, google, auto, status)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.service == 'status':
            show_current_config()
            return
        
        if not args.service:
            # Show current config and available options
            show_current_config()
            available = get_available_services()
            
            if not available:
                print("No AI services configured!")
                print("Please configure at least one API key in your .env file:")
                print("- OPENAI_API_KEY for OpenAI")
                print("- GOOGLE_API_KEY for Google Gemini")
                sys.exit(1)
            
            print("\nUsage:")
            print("  python src/setup.py openai     # Set OpenAI as preferred")
            print("  python src/setup.py google     # Set Google Gemini as preferred")
            print("  python src/setup.py auto       # Enable auto-detection")
            print("  python src/setup.py status     # Show current configuration")
            return
        
        setup_service(args.service)
        
        print("\nNext steps:")
        print("1. Start database: docker-compose up -d")
        print("2. Initialize database: python -m src.config.database")
        print("3. Test system: python -m src.presentation.cli.interface status")
        print("4. Use any CLI command or script - they will use your preferred service")
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
