import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from click.testing import CliRunner
from pathlib import Path
import tempfile
import os

from src.presentation.cli.interface import cli

class TestCLI:
    """Test CLI interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "MBA IA Challenge" in result.output
        assert "PDF Ingestion and Semantic Search" in result.output
    
    def test_cli_version(self):
        """Test CLI version command"""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0.0" in result.output
    
    def test_cli_commands_list(self):
        """Test CLI commands list"""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Commands:" in result.output
        assert "ingest" in result.output
        assert "search" in result.output
        assert "chat" in result.output
        assert "status" in result.output
        assert "validate" in result.output
        assert "list-documents" in result.output

class TestIngestCommand:
    """Test ingest command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._ingest_pdf')
    def test_ingest_command_success(self, mock_ingest):
        """Test successful ingest command"""
        # Mock async function
        mock_ingest.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, ['ingest', tmp_file_path])
            
            assert result.exit_code == 0
            assert "Starting PDF ingestion" in result.output
            assert "Chunk size: 1000" in result.output
            assert "Overlap: 150" in result.output
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_ingest_command_missing_pdf_path(self):
        """Test ingest command without PDF path"""
        result = self.runner.invoke(cli, ['ingest'])
        
        assert result.exit_code != 0
        assert "Missing argument 'PDF_PATH'" in result.output
    
    def test_ingest_command_nonexistent_file(self):
        """Test ingest command with nonexistent file"""
        result = self.runner.invoke(cli, ['ingest', 'nonexistent.pdf'])
        
        assert result.exit_code == 2  # Click validates file existence
        assert "Path 'nonexistent.pdf' does not exist" in result.output
    
    def test_ingest_command_invalid_file_type(self):
        """Test ingest command with invalid file type"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b'Test text content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, ['ingest', tmp_file_path])
            
            assert result.exit_code == 0  # Command doesn't exit with error
            assert "Error: File must be a PDF" in result.output
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_ingest_command_with_options(self):
        """Test ingest command with custom options"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, [
                'ingest',
                tmp_file_path,
                '--chunk-size', '500',
                '--chunk-overlap', '100',
                '--force',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            assert "Chunk size: 500" in result.output
            assert "Overlap: 100" in result.output
            
        finally:
            os.unlink(tmp_file_path)

class TestSearchCommand:
    """Test search command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._search_documents')
    def test_search_command_success(self, mock_search):
        """Test successful search command"""
        # Mock async function
        mock_search.return_value = None
        
        result = self.runner.invoke(cli, ['search', 'What is AI?'])
        
        assert result.exit_code == 0
        assert "Searching for: What is AI?" in result.output
        assert "Results limit: 10" in result.output
    
    def test_search_command_missing_question(self):
        """Test search command without question"""
        result = self.runner.invoke(cli, ['search'])
        
        assert result.exit_code != 0
        assert "Missing argument 'QUESTION'" in result.output
    
    def test_search_command_with_limit(self):
        """Test search command with custom limit"""
        with patch('src.presentation.cli.interface._search_documents') as mock_search:
            mock_search.return_value = None
            
            result = self.runner.invoke(cli, [
                'search',
                'What is AI?',
                '--limit', '5'
            ])
            
            assert result.exit_code == 0
            assert "Results limit: 5" in result.output

class TestChatCommand:
    """Test chat command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._interactive_chat')
    def test_chat_command_startup(self, mock_chat):
        """Test chat command startup"""
        # Mock async function
        mock_chat.return_value = None
        
        result = self.runner.invoke(cli, ['chat'])
        
        assert result.exit_code == 0
        assert "Starting interactive chat mode" in result.output
        assert "Type 'quit', 'exit', or 'q' to exit" in result.output
        assert "Type 'help' for available commands" in result.output

class TestStatusCommand:
    """Test status command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.config.validator.ConfigurationValidator.validate_and_log')
    @patch('src.presentation.cli.interface._show_database_status')
    @patch('src.presentation.cli.interface._show_service_status')
    def test_status_command(self, mock_service_status, mock_db_status, mock_validate):
        """Test status command"""
        # Mock functions
        mock_validate.return_value = None
        mock_db_status.return_value = None
        mock_service_status.return_value = None
        
        result = self.runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        assert "System Status" in result.output
        assert "Configuration: Valid" in result.output
        # Note: Database status is mocked, so we don't check for specific text

class TestValidateCommand:
    """Test validate command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.infrastructure.services.pdf_loader_service.PDFLoaderService.validate_pdf_content')
    def test_validate_command_success(self, mock_validate):
        """Test successful validate command"""
        # Mock validation result
        mock_validate.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {
                'filename': 'test.pdf',
                'file_size': 1024,
                'file_size_mb': 0.001,
                'page_count': 1,
                'total_content_length': 500
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, ['validate', tmp_file_path])
            
            assert result.exit_code == 0
            assert "Validating PDF:" in result.output
            assert "Validation: PASSED" in result.output
            assert "File size: 0.001 MB" in result.output
            assert "Pages: 1" in result.output
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_validate_command_nonexistent_file(self):
        """Test validate command with nonexistent file"""
        result = self.runner.invoke(cli, ['validate', 'nonexistent.pdf'])
        
        assert result.exit_code == 2  # Click validates file existence
        assert "Path 'nonexistent.pdf' does not exist" in result.output

class TestListDocumentsCommand:
    """Test list-documents command"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._list_documents')
    def test_list_documents_command(self, mock_list):
        """Test list-documents command"""
        # Mock async function
        mock_list.return_value = None
        
        result = self.runner.invoke(cli, ['list-documents'])
        
        assert result.exit_code == 0

class TestCLIErrorHandling:
    """Test CLI error handling"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._ingest_pdf')
    def test_ingest_command_error_handling(self, mock_ingest):
        """Test ingest command error handling"""
        # Mock error
        mock_ingest.side_effect = Exception("Test error")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, ['ingest', tmp_file_path])
            
            assert result.exit_code == 0  # Command doesn't exit with error
            assert "Error during ingestion: Test error" in result.output
            
        finally:
            os.unlink(tmp_file_path)
    
    @patch('src.presentation.cli.interface._search_documents')
    def test_search_command_error_handling(self, mock_search):
        """Test search command error handling"""
        # Mock error
        mock_search.side_effect = Exception("Search error")
        
        result = self.runner.invoke(cli, ['search', 'What is AI?'])
        
        assert result.exit_code == 0  # Command doesn't exit with error
        assert "Error during search: Search error" in result.output

class TestCLIVerboseMode:
    """Test CLI verbose mode"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    @patch('src.presentation.cli.interface._ingest_pdf')
    def test_ingest_command_verbose_mode(self, mock_ingest):
        """Test ingest command with verbose mode"""
        # Mock async function
        mock_ingest.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4\n%Test PDF content')
            tmp_file_path = tmp_file.name
        
        try:
            result = self.runner.invoke(cli, [
                'ingest',
                tmp_file_path,
                '--verbose'
            ])
            
            assert result.exit_code == 0
            # Verbose mode should show more detailed output
            
        finally:
            os.unlink(tmp_file_path)
