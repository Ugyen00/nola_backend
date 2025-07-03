import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any
import uuid
import logging
from config import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                # Create index if it doesn't exist
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # text-embedding-3-large dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=config.OPENAI_EMBEDDING_MODEL
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """Store documents in vector database"""
        try:
            # Generate embeddings
            embeddings = self.get_embeddings(texts)
            
            # Create unique IDs for each document
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                # vector = {
                #     "id": ids[i],
                #     "values": embedding,
                #     "metadata": {
                #         "text": text,
                #         **metadata
                #     }
                # }

                vector = {
                    "id": ids[i],
                    "values": embedding,
                    "metadata": {
                        "content": text,
                        "source_id": metadata.get("source_id", ""),
                        "source_type": metadata.get("source_type", ""),
                        "chatbot_id": metadata.get("chatbot_id", "")
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors to Pinecone
            self.index.upsert(vectors=vectors)
            logger.info(f"Stored {len(vectors)} documents in vector database")
            
            return ids
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in Pinecone
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results to match your new structure
            results = []
            for match in search_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("content", ""),  # Get content from metadata
                    "metadata": {
                        "source_id": match.metadata.get("source_id", ""),
                        "source_type": match.metadata.get("source_type", ""),
                        "chatbot_id": match.metadata.get("chatbot_id", "")
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector database")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}