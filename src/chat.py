#!/usr/bin/env python3
"""
Interactive Chat Script

Simple script for interactive chat with the document database.
Usage: python src/chat.py
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.use_cases.search_documents import SearchDocumentsUseCase
from src.config.database import init_database

load_dotenv()


async def interactive_chat():
    """Start interactive chat session."""
    try:
        print("Starting interactive chat mode...")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'help' for available commands")
        print("-" * 50)
        
        # Initialize database
        await init_database()
        
        # Create use case
        use_case = SearchDocumentsUseCase()
        
        while True:
            try:
                # Get user input
                question = input("\nYour question: ").strip()
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Ask any question about your documents")
                    print("  - 'quit', 'exit', or 'q' to exit")
                    print("  - 'help' to show this message")
                    continue
                
                if not question:
                    print("Please enter a question or 'help' for commands")
                    continue
                
                print("Searching...")
                
                # Execute search
                response = await use_case.execute(question=question, limit=10)
                
                print(f"\nResponse:")
                print("-" * 30)
                print(response)
                print("-" * 30)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again or type 'quit' to exit")
        
    except Exception as e:
        print(f"Failed to start chat: {str(e)}")
        sys.exit(1)


def main():
    """Main function to start interactive chat."""
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
