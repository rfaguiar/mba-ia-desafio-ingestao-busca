import click
import asyncio
import logging
from typing import Optional
from pathlib import Path

from src.core.use_cases.ingest_pdf import IngestPDFUseCase
from src.core.use_cases.search_documents import SearchDocumentsUseCase
from src.config.validator import ConfigurationValidator
from src.config.database import init_database
from src.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """MBA IA Challenge - PDF Ingestion and Semantic Search

    A RAG system for ingesting PDF documents and performing semantic search
    using LangChain and PostgreSQL with pgVector.
    """
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--chunk-size", "-c", default=1000, help="Chunk size in characters")
@click.option("--chunk-overlap", "-o", default=150, help="Chunk overlap in characters")
@click.option(
    "--force", "-f", is_flag=True, help="Force re-ingestion even if document exists"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def ingest(
    pdf_path: str, chunk_size: int, chunk_overlap: int, force: bool, verbose: bool
):
    """Ingest PDF document into vector database"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate PDF path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            click.echo(f"Error: PDF file not found: {pdf_path}", err=True)
            return

        if not pdf_file.suffix.lower() == ".pdf":
            click.echo(f"Error: File must be a PDF: {pdf_path}", err=True)
            return

        click.echo(f"Starting PDF ingestion: {pdf_file.name}")
        click.echo(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

        # Run ingestion
        asyncio.run(_ingest_pdf(pdf_path, chunk_size, chunk_overlap, force))

    except Exception as e:
        click.echo(f"Error during ingestion: {str(e)}", err=True)
        if verbose:
            logger.exception("Ingestion failed")


async def _ingest_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int, force: bool):
    """Async PDF ingestion implementation"""
    try:
        # Initialize database
        await init_database()

        # Create use case and execute
        use_case = IngestPDFUseCase()
        result = await use_case.execute(pdf_path, chunk_size, chunk_overlap, force)

        click.echo(f"Successfully ingested {result.total_chunks} chunks")
        click.echo(f"Document ID: {result.id}")
        click.echo(f"File size: {result.file_size} bytes")
        click.echo(f"Pages: {result.page_count}")

    except Exception as e:
        logger.error(f"PDF ingestion failed: {str(e)}")
        raise


@cli.command()
@click.argument("question")
@click.option("--limit", "-l", default=10, help="Number of search results")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def search(question: str, limit: int, verbose: bool):
    """Search documents and get AI-generated response"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        click.echo(f"Searching for: {question}")
        click.echo(f"Results limit: {limit}")

        # Run search
        asyncio.run(_search_documents(question, limit))

    except Exception as e:
        click.echo(f"Error during search: {str(e)}", err=True)
        if verbose:
            logger.exception("Search failed")


async def _search_documents(question: str, limit: int):
    """Async document search implementation"""
    try:
        # Initialize database
        await init_database()

        # Create use case and execute
        use_case = SearchDocumentsUseCase()
        response = await use_case.execute(question, limit)

        click.echo(f"\nResponse: {response}")

    except Exception as e:
        logger.error(f"Document search failed: {str(e)}")
        raise


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def chat(verbose: bool):
    """Interactive chat mode"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        click.echo("Starting interactive chat mode...")
        click.echo("Type 'quit', 'exit', or 'q' to exit")
        click.echo("Type 'help' for available commands")
        click.echo("-" * 50)

        # Run interactive chat
        asyncio.run(_interactive_chat())

    except Exception as e:
        click.echo(f"Error during chat: {str(e)}", err=True)
        if verbose:
            logger.exception("Chat failed")


async def _interactive_chat():
    """Async interactive chat implementation"""
    try:
        # Initialize database
        await init_database()

        # Create use case
        use_case = SearchDocumentsUseCase()

        while True:
            try:
                # Get user input
                question = click.prompt("\nYour question", type=str)

                # Check for exit commands
                if question.lower() in ["quit", "exit", "q"]:
                    click.echo("Goodbye!")
                    break

                # Check for help command
                if question.lower() == "help":
                    _show_chat_help()
                    continue

                # Check for empty input
                if not question.strip():
                    click.echo("Please enter a question.")
                    continue

                # Execute search
                click.echo("Searching...")
                response = await use_case.execute(question)

                click.echo(f"\nResponse: {response}")

            except click.Abort:
                click.echo("\nGoodbye!")
                break
            except Exception as e:
                click.echo(f"Error: {str(e)}", err=True)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception("Chat iteration failed")

    except Exception as e:
        logger.error(f"Interactive chat failed: {str(e)}")
        raise


def _show_chat_help():
    """Show chat help information"""
    help_text = """
Available Commands:
- quit, exit, q: Exit the chat
- help: Show this help message

Tips:
- Ask specific questions for better results
- Questions are answered based on ingested PDF content only
- If information is not in the PDF, you'll get a standard response
"""
    click.echo(help_text)


@cli.command()
def status():
    """Show system status and configuration"""
    try:
        click.echo("System Status")
        click.echo("=" * 50)

        # Validate configuration
        try:
            ConfigurationValidator.validate_and_log()
            click.echo("Configuration: Valid")
        except Exception as e:
            click.echo(f"Configuration: Invalid - {str(e)}")

        # Show database status
        try:
            asyncio.run(_show_database_status())
        except Exception as e:
            click.echo(f"Database: Error - {str(e)}")

        # Show service status
        _show_service_status()
        
        # Show selected AI service
        _show_selected_ai_service()

    except Exception as e:
        click.echo(f"Error getting status: {str(e)}", err=True)


async def _show_database_status():
    """Show database connection status"""
    try:
        # Try to initialize database
        await init_database()
        click.echo("Database: Connected")
    except Exception as e:
        click.echo(f"Database: Not connected - {str(e)}")


def _show_service_status():
    """Show AI service status"""
    try:
        from src.config.settings import settings

        click.echo("\nAI Services:")
        click.echo("-" * 20)

        # OpenAI status
        if (
            settings.OPENAI_API_KEY
            and settings.OPENAI_API_KEY != "your_openai_api_key_here"
        ):
            click.echo(f"OpenAI: Configured ({settings.OPENAI_EMBEDDING_MODEL})")
        else:
            click.echo("OpenAI: Not configured")

        # Gemini status
        if (
            settings.GOOGLE_API_KEY
            and settings.GOOGLE_API_KEY != "your_google_api_key_here"
        ):
            click.echo(f"Gemini: Configured ({settings.GOOGLE_EMBEDDING_MODEL})")
        else:
            click.echo("Gemini: Not configured")

    except Exception as e:
        click.echo(f"Service status: Error - {str(e)}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def validate(pdf_path: str):
    """Validate PDF file before ingestion"""
    try:
        from src.infrastructure.services.pdf_loader_service import PDFLoaderService

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            click.echo(f"Error: PDF file not found: {pdf_path}", err=True)
            return

        click.echo(f"Validating PDF: {pdf_file.name}")

        # Validate PDF
        service = PDFLoaderService()
        validation = service.validate_pdf_content(pdf_path)

        if validation["is_valid"]:
            click.echo("Validation: PASSED")
        else:
            click.echo("Validation: FAILED")

        # Show file info
        file_info = validation["file_info"]
        click.echo(f"File size: {file_info['file_size_mb']} MB")
        click.echo(f"Pages: {file_info.get('page_count', 'Unknown')}")
        click.echo(
            f"Total content length: {file_info.get('total_content_length', 'Unknown')}"
        )

        # Show warnings
        if validation["warnings"]:
            click.echo("\nWarnings:")
            for warning in validation["warnings"]:
                click.echo(f"  - {warning}")

        # Show errors
        if validation["errors"]:
            click.echo("\nErrors:")
            for error in validation["errors"]:
                click.echo(f"  - {error}")

    except Exception as e:
        click.echo(f"Error during validation: {str(e)}", err=True)


@cli.command()
def list_documents():
    """List all ingested documents"""
    try:
        asyncio.run(_list_documents())
    except Exception as e:
        click.echo(f"Error listing documents: {str(e)}", err=True)


async def _list_documents():
    """Async document listing implementation"""
    try:
        # Initialize database
        await init_database()

        # Get documents summary
        from src.infrastructure.database.repositories.postgres_document_repository import (
            PostgresDocumentRepository,
        )
        from src.config.database import get_async_session

        async for session in get_async_session():
            repository = PostgresDocumentRepository(session)
            documents = await repository.get_documents_summary()

            if not documents:
                click.echo("No documents found in database.")
                return

            click.echo(f"Found {len(documents)} documents:")
            click.echo("-" * 80)

            for doc in documents:
                click.echo(f"ID: {doc['id']}")
                click.echo(f"Filename: {doc['filename']}")
                click.echo(f"Chunks: {doc['total_chunks']}")
                click.echo(f"Size: {doc['file_size']} bytes")
                click.echo(f"Pages: {doc['page_count']}")
                click.echo(f"Created: {doc['created_at']}")
                click.echo("-" * 40)

    except Exception as e:
        logger.error(f"Document listing failed: {str(e)}")
        raise


def _show_selected_ai_service():
    """Show the currently selected AI service."""
    try:
        settings = Settings()
        service_info = settings.ai_service_info
        
        click.echo("\nAI Service Selection:")
        click.echo("-" * 30)
        
        if service_info['selected_service']:
            click.echo(f"Selected Service: {service_info['selected_service'].upper()}")
            
            if service_info['preferred_service']:
                click.echo(f"Preferred Service: {service_info['preferred_service'].upper()}")
            else:
                click.echo("Preferred Service: Auto-detect")
        else:
            click.echo("Selected Service: None (no services configured)")
        
        click.echo(f"Available Services: {', '.join(service_info['available_services']).upper() or 'None'}")
        
        if not service_info['available_services']:
            click.echo("\nTo configure AI services, run: python src/setup.py")
        
    except Exception as e:
        click.echo(f"Error showing AI service status: {str(e)}")


if __name__ == "__main__":
    cli()
