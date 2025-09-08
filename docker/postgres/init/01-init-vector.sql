-- Initialize pgVector extension
-- This script ensures pgVector is always available in the database

-- Create the vector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'pgVector extension initialized successfully';
END $$;
