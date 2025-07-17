from typing import List, Tuple
from .database import create_connection, search_similar_chunks
from .embeddings import EmbeddingGenerator

class SemanticSearcher:
    def __init__(self):
        """Initialize the semantic searcher with database connection and embedding generator"""
        self.conn = create_connection()
        self.embedding_generator = EmbeddingGenerator()
    
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
        # Generate embedding for the query
        query_embedding = self.embedding_generator.get_embedding(query)
        
        # Search for similar chunks
        results = search_similar_chunks(self.conn, query_embedding, limit)
        
        # Process results
        processed_results = []
        for filename, chunk_text, similarity in results:
            # Parse the chunk text
            parts = chunk_text.split('\n')
            question = ''
            answer = ''
            content = chunk_text
            category = filename.split(' - ')[0] if ' - ' in filename else 'Unknown'
            
            # Extract question and answer if available
            for part in parts:
                if part.startswith('Question: '):
                    question = part.replace('Question: ', '')
                elif part.startswith('Answer: '):
                    answer = part.replace('Answer: ', '')
            
            processed_results.append({
                'question': question,
                'answer': answer,
                'content': content,
                'similarity': float(similarity),
                'category': category
            })
        
        return processed_results

def format_search_result(result: dict) -> str:
    """Format a search result for display"""
    formatted = f"Category: {result['category']}\n"
    if result['question']:
        formatted += f"Q: {result['question']}\n"
    if result['answer']:
        formatted += f"A: {result['answer']}\n"
    formatted += f"\nRelevance Score: {result['similarity']:.2%}"
    return formatted 