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
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

def prepare_text_for_embedding(question: str, answer: str, content: str) -> str:
    """Prepare text for embedding by combining question, answer, and content"""
    return f"Question: {question}\nAnswer: {answer}\nContent: {content}" 