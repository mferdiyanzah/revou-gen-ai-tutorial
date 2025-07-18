from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client with optional API key override"""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        try:
            print(f"Getting embedding for text: {text[:100]}...")
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            print(f"Successfully generated embedding of length {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to handle in caller
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call"""
        try:
            print(f"Getting batch embeddings for {len(texts)} texts...")
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            print(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to handle in caller

def prepare_text_for_embedding(question: str, answer: str) -> str:
    """Prepare text for embedding by combining question, answer"""
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}" 