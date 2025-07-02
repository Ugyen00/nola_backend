from typing import Dict, Any, List
import logging
from document_processor import DocumentProcessor
from vector_store import VectorStore
from models import DocumentType

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
    
    async def train_from_document(self, document_type: DocumentType, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the chatbot with a new document"""
        try:
            if metadata is None:
                metadata = {}
            
            # Process document based on type
            documents = await self.document_processor.process_document(
                document_type.value, 
                content, 
                metadata
            )
            
            if not documents:
                return {
                    "success": False,
                    "message": f"Failed to process {document_type.value} document",
                    "chunks_processed": 0
                }
            
            # Extract texts and metadata for vector storage
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Store in vector database
            document_ids = self.vector_store.store_documents(texts, metadatas)
            
            logger.info(f"Successfully trained with {len(documents)} chunks from {document_type.value}")
            
            return {
                "success": True,
                "message": f"Successfully processed {document_type.value} document",
                "document_ids": document_ids,
                "chunks_processed": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error training from {document_type.value}: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {document_type.value}: {str(e)}",
                "chunks_processed": 0
            }
    
    async def batch_train(self, training_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train with multiple documents in batch"""
        results = []
        total_chunks = 0
        successful_docs = 0
        
        for request in training_requests:
            try:
                document_type = DocumentType(request.get("document_type"))
                content = request.get("content")
                metadata = request.get("metadata", {})
                
                result = await self.train_from_document(document_type, content, metadata)
                results.append(result)
                
                if result["success"]:
                    successful_docs += 1
                    total_chunks += result["chunks_processed"]
                    
            except Exception as e:
                logger.error(f"Error in batch training: {str(e)}")
                results.append({
                    "success": False,
                    "message": f"Error processing document: {str(e)}",
                    "chunks_processed": 0
                })
        
        return {
            "success": successful_docs > 0,
            "message": f"Batch training completed. {successful_docs}/{len(training_requests)} documents processed successfully.",
            "total_chunks_processed": total_chunks,
            "successful_documents": successful_docs,
            "failed_documents": len(training_requests) - successful_docs,
            "detailed_results": results
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics from vector store"""
        try:
            stats = self.vector_store.get_index_stats()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting training stats: {str(e)}")
            return {
                "success": False,
                "message": f"Error retrieving stats: {str(e)}"
            }
    
    def delete_documents_by_source(self, source: str) -> Dict[str, Any]:
        """Delete all documents from a specific source"""
        try:
            # This would require implementing a metadata filter search in vector_store
            # For now, return a placeholder response
            return {
                "success": False,
                "message": "Document deletion by source not yet implemented"
            }
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return {
                "success": False,
                "message": f"Error deleting documents: {str(e)}"
            }