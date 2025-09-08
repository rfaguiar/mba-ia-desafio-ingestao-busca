from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict, Any, Optional
import os
import logging
from src.core.domain.document import Document
from src.core.domain.chunk import Chunk

logger = logging.getLogger(__name__)


class PDFLoaderService:
    """Service for loading and processing PDF documents"""

    def __init__(self):
        self.supported_extensions = [".pdf"]

    def can_load_file(self, file_path: str) -> bool:
        """Check if file can be loaded by this service"""
        if not os.path.exists(file_path):
            return False

        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_extensions

    def load_pdf(self, file_path: str) -> Document:
        """Load PDF file and return Document entity"""
        if not self.can_load_file(file_path):
            raise ValueError(f"Cannot load file: {file_path}")

        try:
            logger.info(f"Loading PDF file: {file_path}")

            # Get file information
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)

            # Load PDF using LangChain
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Create document entity
            document = Document(
                filename=filename, file_size=file_size, page_count=len(pages)
            )

            logger.info(
                f"Successfully loaded PDF: {filename} ({len(pages)} pages, {file_size} bytes)"
            )

            return document

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def extract_text_from_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from all pages with metadata"""
        if not self.can_load_file(file_path):
            raise ValueError(f"Cannot load file: {file_path}")

        try:
            logger.info(f"Extracting text from PDF: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            extracted_pages = []
            for i, page in enumerate(pages):
                page_data = {
                    "page_number": i + 1,
                    "content": page.page_content,
                    "metadata": page.metadata,
                    "content_length": len(page.page_content),
                }
                extracted_pages.append(page_data)

            logger.info(f"Extracted text from {len(extracted_pages)} pages")
            return extracted_pages

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def validate_pdf_content(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF content and return validation results"""
        if not self.can_load_file(file_path):
            raise ValueError(f"Cannot load file: {file_path}")

        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "file_info": {},
            }

            # Get file info
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)

            validation_result["file_info"] = {
                "filename": filename,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
            }

            # Check file size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                validation_result["warnings"].append("File size is large (>100MB)")

            # Try to load PDF
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                validation_result["file_info"]["page_count"] = len(pages)
                validation_result["file_info"]["total_content_length"] = sum(
                    len(page.page_content) for page in pages
                )

                # Check for empty pages
                empty_pages = [
                    i + 1
                    for i, page in enumerate(pages)
                    if not page.page_content.strip()
                ]
                if empty_pages:
                    validation_result["warnings"].append(
                        f"Empty pages found: {empty_pages}"
                    )

                # Check for very short content
                short_pages = [
                    i + 1
                    for i, page in enumerate(pages)
                    if len(page.page_content.strip()) < 50
                ]
                if short_pages:
                    validation_result["warnings"].append(
                        f"Pages with very short content: {short_pages}"
                    )

            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Failed to load PDF: {str(e)}")

            logger.info(f"PDF validation completed: {filename}")
            return validation_result

        except Exception as e:
            logger.error(f"Error validating PDF {file_path}: {str(e)}")
            raise
