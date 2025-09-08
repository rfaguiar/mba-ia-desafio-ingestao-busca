#!/usr/bin/env python3
"""
PDF Ingestion Script

Simple script to ingest PDF documents into the vector database.
Usage: python src/ingest.py <pdf_path> [--chunk-size 1000] [--chunk-overlap 150]
"""

import asyncio
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.use_cases.ingest_pdf import IngestPDFUseCase
from src.config.database import init_database

load_dotenv()


async def ingest_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Ingest a PDF document into the vector database."""
    try:
        print(f"Starting PDF ingestion: {pdf_path}")
        print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        # Initialize database
        await init_database()
        
        # Create use case
        use_case = IngestPDFUseCase()
        
        # Execute ingestion
        document = await use_case.execute(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        print(f"Successfully ingested document: {document.filename}")
        print(f"Pages: {document.page_count}")
        print(f"Chunks: {document.total_chunks}")
        print(f"Size: {document.file_size} bytes")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF document into vector database"
    )
    parser.add_argument(
        "pdf_path", 
        help="Path to the PDF file to ingest"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=150, 
        help="Chunk overlap in characters (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if not pdf_file.suffix.lower() == '.pdf':
        print(f"Error: File must be a PDF: {args.pdf_path}")
        sys.exit(1)
    
    # Run ingestion
    asyncio.run(ingest_pdf(args.pdf_path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
