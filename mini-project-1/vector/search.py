from typing import List, Tuple
from .database import create_connection, search_similar_chunks
from .embeddings import EmbeddingGenerator
import psycopg2

class SemanticSearcher:
    def __init__(self):
        """Initialize the semantic searcher with embedding generator"""
        self.conn = None
        self.embedding_generator = EmbeddingGenerator()
    
    def _ensure_connection(self):
        """Ensure we have a valid database connection"""
        if self.conn is None or self.conn.closed:
            print("Creating new database connection...")
            self.conn = create_connection()
        else:
            # Test the connection by executing a simple query
            try:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            except (psycopg2.InterfaceError, psycopg2.OperationalError):
                print("Connection is stale, creating new connection...")
                self.conn = create_connection()
    
    def search(self, query: str, limit: int = 5) -> List[dict]:
        """
        Search for relevant chunks based on semantic similarity
        
        Args:
            query: The search query (question) from the user
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing:
            - question: The original question from the chunk
            - answer: The answer from the chunk
            - content: The full content of the chunk
            - similarity: The similarity score (0-1, higher is better)
            - category: The document category
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure we have a valid connection
                self._ensure_connection()
                
                # Generate embedding for the query
                print(f"Generating embedding for query: {query}")
                query_embedding = self.embedding_generator.get_embedding(query)
                
                # Search for similar chunks
                print("Searching for similar chunks...")
                results = search_similar_chunks(self.conn, query_embedding, limit)
                print(f"Found {len(results)} results from database")
                
                # Process results
                processed_results = []
                for filename, chunk_text, similarity in results:
                    print(f"Processing result from {filename} with similarity {similarity}")
                    
                    # Parse the chunk text
                    parts = chunk_text.split('\n')
                    question = ''
                    answer = ''
                    content = chunk_text
                    category = filename.split(' - ')[0] if ' - ' in filename else 'Unknown'
                    
                    # Extract question and answer if available
                    question_found = False
                    answer_found = False
                    answer_lines = []
                    
                    for i, part in enumerate(parts):
                        if part.startswith('Question: '):
                            question = part.replace('Question: ', '')
                            question_found = True
                        elif part.startswith('Answer: '):
                            answer_found = True
                            # Start collecting answer from this line
                            answer_lines.append(part.replace('Answer: ', ''))
                            # Continue collecting subsequent lines until we hit another section or end
                            for j in range(i + 1, len(parts)):
                                next_part = parts[j].strip()
                                # Stop if we hit another Question or if we reach content that looks like metadata
                                if (next_part.startswith('Question: ') or 
                                    next_part.startswith('Nama "Dexa"') or
                                    next_part.startswith('Source:') or
                                    next_part.startswith('Category:')):
                                    break
                                # Include the line if it's not empty or if it's a bullet point
                                if next_part or (j < len(parts) - 1 and parts[j + 1].strip().startswith('o')):
                                    answer_lines.append(parts[j])
                            break
                    
                    # Join answer lines and clean up
                    if answer_lines:
                        answer = '\n'.join(answer_lines).strip()
                        # Clean up any trailing metadata
                        if 'Nama "Dexa"' in answer:
                            answer = answer.split('Nama "Dexa"')[0].strip()
                    else:
                        answer = ''
                    
                    processed_result = {
                        'question': question,
                        'answer': answer,
                        'content': content,
                        'similarity': float(similarity),
                        'category': category
                    }
                    processed_results.append(processed_result)
                    print(f"Processed result: Q: {question[:50]}... A: {answer[:50]}...")
                
                return processed_results
                
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as db_error:
                print(f"Database connection error (attempt {attempt + 1}/{max_retries}): {str(db_error)}")
                # Force connection recreation on next attempt
                self.conn = None
                if attempt == max_retries - 1:
                    print("Max retries reached for database connection")
                    raise
                continue
            except Exception as e:
                print(f"Error in semantic search: {str(e)}")
                import traceback
                traceback.print_exc()
                return []
    
    def close(self):
        """Close the database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None

def format_search_result(result: dict) -> str:
    """Format a search result for display"""
    formatted = f"Category: {result['category']}\n"
    if result['question']:
        formatted += f"Q: {result['question']}\n"
    if result['answer']:
        formatted += f"A: {result['answer']}\n"
    formatted += f"\nRelevance Score: {result['similarity']:.2%}"
    return formatted 