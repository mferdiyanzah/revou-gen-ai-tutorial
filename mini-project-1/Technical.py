import streamlit as st
from urllib.error import URLError
import io
import json
from vector import (
    create_connection,
    create_tables,
    batch_insert_chunks,
    EmbeddingGenerator,
    prepare_text_for_embedding,
    chunk_exists
)
import os
from dotenv import load_dotenv

# Import functions from docs.doc_functions module following GeeksforGeeks pattern
from docs.doc_functions import doc_extractor, doc_mapper, create_question_chunks, get_chunk_summaries

# Load environment variables
load_dotenv()

def prepare_chunks_for_db(chunks: list, embedding_generator: EmbeddingGenerator, conn) -> tuple[list, int]:
    """Prepare chunks for database insertion with embeddings"""
    db_chunks = []
    skipped_count = 0
    
    for chunk in chunks:
        # Create a meaningful filename from category and question
        filename = f"{chunk['category']} - {chunk['question'][:100]}" if chunk['question'] else f"{chunk['category']} - General"
        
        # Check if chunk already exists before generating embedding
        if chunk_exists(conn, filename, chunk['chunk_id']):
            skipped_count += 1
            continue
            
        # Combine relevant text for embedding
        text_for_embedding = prepare_text_for_embedding(
            question=chunk['question'],
            answer=chunk['answer'],
        )
        
        # Get embedding
        try:
            embedding = embedding_generator.get_embedding(text_for_embedding)
            
            # Prepare chunk for database
            db_chunks.append((
                filename,
                chunk['chunk_id'],
                text_for_embedding,
                embedding
            ))
        except Exception as e:
            st.warning(f"Failed to get embedding for chunk {chunk['chunk_id']}: {str(e)}")
            continue
            
    return db_chunks, skipped_count

try:
    st.title("FAQ Dexa Medica - Question-Based Chunking")
    
    # Extract and map document
    extracted_text = doc_extractor()
    doc_map = doc_mapper(extracted_text)
    
    # Create question-based chunks
    chunks = create_question_chunks(doc_map)
    chunk_summary = get_chunk_summaries(chunks)
    
    # Display summary
    st.header("📊 Chunk Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Chunks", chunk_summary['total_chunks'])
    
    with col2:
        st.metric("Total Categories", len(chunk_summary['categories']))
    
    # Display chunks per category
    st.subheader("Questions per Category")
    for category, count in chunk_summary['categories'].items():
        st.write(f"**{category}**: {count} questions")
    
    # Show chunks
    st.header("📝 Question Chunks")
    
    # Category filter
    selected_category = st.selectbox(
        "Filter by Category",
        ["All"] + list(chunk_summary['categories'].keys())
    )
    
    # Filter chunks based on selection
    filtered_chunks = chunks
    if selected_category != "All":
        filtered_chunks = [chunk for chunk in chunks if chunk['category'] == selected_category]
    
    # Display filtered chunks
    for i, chunk in enumerate(filtered_chunks):
        with st.expander(f"Chunk {chunk['chunk_id']}: {chunk['question'][:60]}..." if chunk['question'] else f"Chunk {chunk['chunk_id']}: {chunk['category']}"):
            st.write("**Category:**", chunk['category'])
            if chunk['question']:
                st.write("**Question:**", chunk['question'])
                st.write("**Answer:**", chunk['answer'])
            st.write("**Full Content:**")
            st.text(chunk['content'])
            st.write("**Metadata:**", chunk['metadata'])
    
    # Database Operations Section
    st.header("🗄️ Database Operations")
    
    # Initialize database
    if st.button("Initialize Database"):
        try:
            conn = create_connection()
            create_tables(conn)
            st.success("Database initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize database: {e}")
    
    # Store chunks in database
    if st.button("Store Chunks in Database"):
        try:
            # Create connection and embedding generator
            conn = create_connection()
            embedding_generator = EmbeddingGenerator()
            
            # Prepare chunks with embeddings
            with st.spinner("Preparing chunks and generating embeddings..."):
                db_chunks, skipped_count = prepare_chunks_for_db(chunks, embedding_generator, conn)
            
            # Insert chunks
            with st.spinner("Storing chunks in database..."):
                inserted_count = batch_insert_chunks(conn, db_chunks)
                
            st.success(f"Successfully stored {inserted_count} new chunks in the database!")
            
            if skipped_count > 0:
                st.info(f"Skipped {skipped_count} chunks that were already in the database.")
                
        except Exception as e:
            st.error(f"Failed to store chunks: {e}")
        
except URLError as e:
    st.error(e.reason)