import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"  # Changed from gpt-4-turbo-preview to gpt-4o-mini
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # Best OpenAI embedding model
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot-index")
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # API Configuration
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7

config = Config()