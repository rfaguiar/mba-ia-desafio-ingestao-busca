# PDF Search System - Easy Document Search with AI

A simple system that lets you upload PDF documents and ask questions about their content using artificial intelligence.

## Features

- **PDF Processing**: Load, validate, and chunk PDF documents
- **AI Integration**: Support for OpenAI and Google Gemini models
- **Vector Storage**: PostgreSQL with pgVector extension for embeddings
- **Semantic Search**: Find relevant document chunks using vector similarity
- **RAG Pipeline**: Generate contextual responses using retrieved content
- **CLI Interface**: Command-line tools for all operations

## Prerequisites

- Python 3.11+
- PostgreSQL 13+ with pgVector extension
- Docker and Docker Compose
- OpenAI API key or Google Gemini API key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mba-ia-desafio-ingestao-busca
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file with your API keys:

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag

# OpenAI (choose one)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5-nano

# Google Gemini (alternative)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_LLM_MODEL=gemini-2.5-flash-lite
```

### 5. Configure AI Service

```bash
# Set Google Gemini as preferred service
python src/setup.py google

# Or set OpenAI
python src/setup.py openai

# Or enable auto-detection
python src/setup.py auto
```

### 6. Start Database

```bash
docker-compose up -d
```

### 7. Initialize Database

```bash
export PYTHONPATH=$(pwd)
python -m src.config.database
```

## Usage

### Step 1: Check System Status

```bash
export PYTHONPATH=$(pwd)
python -m src.presentation.cli.interface status
```

### Step 2: Ingest PDF Documents

```bash
# Basic ingestion
python src/ingest.py document.pdf

# With custom chunk size
python src/ingest.py document.pdf --chunk-size 500 --chunk-overlap 100

# Using CLI command
python -m src.presentation.cli.interface ingest document.pdf
```

### Step 3: Search Documents

```bash
# Search for specific information
python src/search.py "What companies are mentioned in the document?"

# Search for financial data
python src/search.py "What is the revenue of Company X?"

# Search with filters
python src/search.py "What companies have revenue over 100 million?" --limit 5

# Search for specific company revenue
python src/search.py "Qual o faturamento da Empresa SuperTechIABrazil?"

# Search for information not in context
python src/search.py "Quantos clientes temos em 2024?"

# Using CLI command
python -m src.presentation.cli.interface search "What companies are mentioned?"
```

### Step 4: Interactive Chat

```bash
# Start interactive chat session
python src/chat.py

# Using CLI command
python -m src.presentation.cli.interface chat
```

### Step 5: Manage Documents

```bash
# List all ingested documents
python -m src.presentation.cli.interface list-documents

# Validate PDF before ingestion
python -m src.presentation.cli.interface validate document.pdf
```

## CLI Commands Reference

```bash
# Show all available commands
python -m src.presentation.cli.interface --help

# Check system status
python -m src.presentation.cli.interface status

# List ingested documents
python -m src.presentation.cli.interface list-documents

# Validate PDF before ingestion
python -m src.presentation.cli.interface validate document.pdf

# Ingest PDF document
python -m src.presentation.cli.interface ingest document.pdf

# Search documents
python -m src.presentation.cli.interface search "your question"

# Start interactive chat
python -m src.presentation.cli.interface chat
```

## Example Questions

Try these types of questions for best results:

- "What companies are mentioned in the document?"
- "What is the revenue of [specific company]?"
- "What companies have revenue over [amount]?"
- "What companies are in the [sector] sector?"
- "What type of data is in this document?"
- "List all companies with revenue between X and Y"

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'src'**
   ```bash
   # Solution: Set PYTHONPATH before running commands
   export PYTHONPATH=$(pwd)
   ```

2. **Database Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps
   
   # Restart database
   docker-compose restart postgres
   ```

3. **API Key Errors**
   ```bash
   # Verify environment variables
   python -m src.presentation.cli.interface status
   
   # Check if .env file exists and has correct keys
   cat .env | grep API_KEY
   ```

4. **No Documents Found**
   ```bash
   # Check if documents were ingested
   python -m src.presentation.cli.interface list-documents
   
   # Re-ingest documents if needed
   python src/ingest.py document.pdf
   ```

### Getting Help

1. **Check System Status**: Always run `python -m src.presentation.cli.interface status` first
2. **Validate Configuration**: Ensure your `.env` file has valid API keys
3. **Test with Small Files**: Start with small PDF files to test the system
4. **Set PYTHONPATH**: Make sure to set `export PYTHONPATH=$(pwd)` before running any Python commands

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Performance tests
pytest tests/performance/ -v
```

## Configuration

### Chunking Parameters

- **CHUNK_SIZE**: Maximum characters per chunk (default: 1000)
- **CHUNK_OVERLAP**: Overlap between consecutive chunks (default: 200)
- **SIMILARITY_THRESHOLD**: Minimum similarity for search results (default: 0.0)

### AI Model Settings

- **Embedding Models**: OpenAI text-embedding-3-small, Google models/embedding-001
- **LLM Models**: OpenAI gpt-5-nano, Google gemini-2.5-flash-lite

## Architecture

This project follows Clean Architecture principles:

```
src/
├── chat.py                 # Interactive chat interface
├── ingest.py               # PDF document ingestion script
├── search.py               # Document search and retrieval script
├── setup.py                # LLM service configuration script
├── config/                 # Configuration management
├── core/                   # Business logic and domain entities
│   ├── domain/             # Domain models (Document, Chunk)
│   ├── interfaces/         # Abstract interfaces
│   └── use_cases/          # Business use cases
├── infrastructure/         # External dependencies
│   ├── database/           # Database models and repositories
│   ├── services/           # AI service implementations
│   └── text_processing/    # PDF and text processing
└── presentation/           # User interface (CLI)
    ├── cli/                # Command-line interface components
    └── prompts/            # AI prompt templates
```

## License

This project is licensed under the MIT License.

## Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the test examples