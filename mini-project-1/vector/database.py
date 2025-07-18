import os
import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import numpy as np
from typing import List, Optional, Tuple

# Load environment variables from .env file
load_dotenv()

__all__ = [
    'create_connection',
    'create_tables',
    'chunk_exists',
    'insert_chunk',
    'batch_insert_chunks',
    'search_similar_chunks',
    'get_all_chunks'
]

def init_vector_extension(conn: connection):
    """Initialize the pgvector extension if it doesn't exist"""

    # check if vector extension is already created
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            return

    with conn.cursor() as cur:
        # Create extension if it doesn't exist
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

def create_connection() -> connection:
    """Create a database connection with pgvector support"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'dexa_medica'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '123456')
    )

    # Initialize vector extension
    init_vector_extension(conn)
    
    return conn

def create_tables(conn: connection):
    """Create necessary tables for storing document chunks and embeddings"""
    with conn.cursor() as cur:
        # Create documents table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(1536),  -- OpenAI embedding dimension
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(filename, chunk_index)
        );
        """)
        
        # Create an index on the embedding column for faster similarity search
        cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """)
        
        conn.commit()

def chunk_exists(conn: connection, filename: str, chunk_index: int) -> bool:
    """Check if a specific chunk already exists in the database"""
    with conn.cursor() as cur:
        cur.execute("""
        SELECT EXISTS(
            SELECT 1 
            FROM documents 
            WHERE filename = %s AND chunk_index = %s
        );
        """, (filename, chunk_index))
        result = cur.fetchone()
        return bool(result[0]) if result else False

def insert_chunk(
    conn: connection,
    filename: str,
    chunk_index: int,
    chunk_text: str,
    embedding: Optional[List[float]] = None
) -> bool:
    """
    Insert a document chunk and its embedding into the database
    Returns True if insertion was successful, False if chunk already exists
    """
    if chunk_exists(conn, filename, chunk_index):
        return False
    
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO documents (filename, chunk_index, chunk_text, embedding)
        VALUES (%s, %s, %s, %s)
        """, (filename, chunk_index, chunk_text, embedding))
        conn.commit()
    return True

def batch_insert_chunks(
    conn: connection,
    chunks: List[Tuple[str, int, str, List[float]]]
) -> int:
    """
    Batch insert multiple chunks and their embeddings
    Returns the number of chunks successfully inserted
    """
    inserted_count = 0
    with conn.cursor() as cur:
        # Filter out existing chunks
        new_chunks = [
            chunk for chunk in chunks 
            if not chunk_exists(conn, chunk[0], chunk[1])
        ]
        
        if new_chunks:
            execute_values(
                cur,
                """
                INSERT INTO documents (filename, chunk_index, chunk_text, embedding)
                VALUES %s
                """,
                new_chunks
            )
            conn.commit()
            inserted_count = len(new_chunks)
    
    return inserted_count

def search_similar_chunks(
    conn: connection,
    query_embedding: List[float],
    limit: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Search for similar chunks using cosine similarity
    Returns list of (filename, chunk_text, similarity_score)
    """
    with conn.cursor() as cur:
        cur.execute("""
        SELECT filename, chunk_text, 1 - (embedding <=> %s::vector) as similarity
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT %s
        """, (query_embedding, limit))  # Fixed: removed duplicate parameter
        
        results = cur.fetchall()
        print('------------------------------------')
        print('Search results:', results)
        return results

def get_all_chunks(conn: connection, filename: Optional[str] = None) -> List[Tuple[str, int, str]]:
    """
    Retrieve all chunks, optionally filtered by filename
    Returns list of (filename, chunk_index, chunk_text)
    """
    with conn.cursor() as cur:
        if filename:
            cur.execute("""
            SELECT filename, chunk_index, chunk_text
            FROM documents
            WHERE filename = %s
            ORDER BY chunk_index
            """, (filename,))
        else:
            cur.execute("""
            SELECT filename, chunk_index, chunk_text
            FROM documents
            ORDER BY filename, chunk_index
            """)
        
        return cur.fetchall()

