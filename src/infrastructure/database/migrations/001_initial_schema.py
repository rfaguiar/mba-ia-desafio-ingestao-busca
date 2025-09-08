"""
Initial database schema migration
Creates documents, chunks, and search_history tables with pgVector support
Supports dynamic embedding dimensions for both OpenAI (1536) and Google Gemini (768)
"""

import asyncio
from sqlalchemy import text
from src.config.database import async_engine
from src.config.settings import Settings

settings = Settings()


def get_embedding_dimension() -> int:
    """Get the appropriate embedding dimension based on preferred service"""
    try:
        with open('.preferred_service', 'r') as f:
            preferred_service = f.read().strip().lower()
        
        if preferred_service == 'google':
            return 768   # Google Gemini embedding dimension
        elif preferred_service == 'openai':
            return 1536  # OpenAI embedding dimension
        else:
            # Fallback to checking available keys
            if settings.OPENAI_API_KEY:
                return 1536  # OpenAI embedding dimension
            elif settings.GOOGLE_API_KEY:
                return 768   # Google Gemini embedding dimension
            else:
                return 1536  # Default to OpenAI dimension
    except FileNotFoundError:
        # Fallback to checking available keys
        if settings.OPENAI_API_KEY:
            return 1536  # OpenAI embedding dimension
        elif settings.GOOGLE_API_KEY:
            return 768   # Google Gemini embedding dimension
        else:
            return 1536  # Default to OpenAI dimension


async def upgrade():
    """Create initial database schema with dynamic embedding dimensions"""
    embedding_dim = get_embedding_dimension()
    print(f"Creating database schema with {embedding_dim} dimension embeddings")
    
    async with async_engine.begin() as conn:
        # Create pgVector extension if it doesn't exist
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # Create documents table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                content_hash VARCHAR(64) UNIQUE NOT NULL,
                total_chunks INTEGER DEFAULT 0,
                file_size INTEGER,
                page_count INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE
            );
        """))

        # Create chunks table with dynamic embedding dimension
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                embedding vector({embedding_dim}) NOT NULL,
                chunk_metadata TEXT,
                page_number INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))

        # Create search_history table with dynamic embedding dimension
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS search_history (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                query_embedding vector({embedding_dim}) NOT NULL,
                response TEXT NOT NULL,
                search_results_count INTEGER DEFAULT 0,
                processing_time_ms INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """))

        # Create basic indexes
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_document_chunk_index ON chunks(document_id, chunk_index);"))

        # Create vector indexes for performance
        await conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
            ON chunks USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """))

        await conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_search_history_embedding 
            ON search_history USING ivfflat (query_embedding vector_cosine_ops) 
            WITH (lists = 100);
        """))

        print(f"Database schema created successfully with {embedding_dim} dimension embeddings")


async def downgrade():
    """Drop all tables"""
    async with async_engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS search_history CASCADE;"))
        await conn.execute(text("DROP TABLE IF EXISTS documents CASCADE;"))
        print("Database schema dropped successfully")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        asyncio.run(downgrade())
    else:
        asyncio.run(upgrade())
