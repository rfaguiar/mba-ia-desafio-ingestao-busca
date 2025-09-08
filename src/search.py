#!/usr/bin/env python3
"""
Document Search Script

Simple script to search documents and get AI-generated responses.
Usage: python src/search.py "<question>" [--limit 10]
"""

import asyncio
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.use_cases.search_documents import SearchDocumentsUseCase
from src.config.database import init_database

load_dotenv()


async def search_documents(question: str, limit: int = 10):
    """Search documents and get AI-generated response."""
    try:
        print(f"Searching for: {question}")
        print(f"Results limit: {limit}")
        
        # Initialize database
        await init_database()
        
        # Create use case
        use_case = SearchDocumentsUseCase()
        
        # Execute search
        response = await use_case.execute(
            question=question,
            limit=limit
        )
        
        print(f"\nAI Response:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search documents and get AI-generated response"
    )
    parser.add_argument(
        "question", 
        help="Question to ask about the documents"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10, 
        help="Number of search results to consider (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate question
    if not args.question.strip():
        print("Error: Question cannot be empty")
        sys.exit(1)
    
    # Run search
    asyncio.run(search_documents(args.question, args.limit))


if __name__ == "__main__":
    main()