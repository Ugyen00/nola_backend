# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# import logging
# import uvicorn
# from typing import List, Optional
# import base64

# # Import our custom modules
# from models import (
#     TrainingRequest, 
#     ChatRequest, 
#     ChatResponse, 
#     TrainingResponse, 
#     HealthResponse,
#     DocumentType
# )
# from training_service import TrainingService
# from chat_service import ChatService
# from config import config

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="RAG AI Chatbot Backend",
#     description="A FastAPI backend for RAG-based AI chatbot with training capabilities",
#     version="1.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize services
# training_service = TrainingService()
# chat_service = ChatService()

# @app.get("/", response_model=HealthResponse)
# async def root():
#     """Root endpoint"""
#     return HealthResponse(
#         status="healthy",
#         message="RAG AI Chatbot Backend is running"
#     )

# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     try:
#         # You could add more sophisticated health checks here
#         # like checking database connections, etc.
#         return HealthResponse(
#             status="healthy",
#             message="All services are operational"
#         )
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="Service unhealthy")

# @app.post("/train", response_model=TrainingResponse)
# async def train_chatbot(request: TrainingRequest):
#     """Train the chatbot with new content"""
#     try:
#         logger.info(f"Training request received for {request.document_type}")
        
#         result = await training_service.train_from_document(
#             request.document_type,
#             request.content,
#             request.metadata
#         )
        
#         if result["success"]:
#             return TrainingResponse(
#                 success=True,
#                 message=result["message"],
#                 chunks_processed=result["chunks_processed"]
#             )
#         else:
#             raise HTTPException(status_code=400, detail=result["message"])
            
#     except Exception as e:
#         logger.error(f"Training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# @app.post("/train/url")
# async def train_from_url(url: str = Form(...), metadata: Optional[str] = Form(None)):
#     """Train from website URL"""
#     try:
#         # Parse metadata if provided
#         import json
#         parsed_metadata = json.loads(metadata) if metadata else {}
        
#         request = TrainingRequest(
#             document_type=DocumentType.URL,
#             content=url,
#             metadata=parsed_metadata
#         )
        
#         return await train_chatbot(request)
        
#     except Exception as e:
#         logger.error(f"URL training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"URL training failed: {str(e)}")

# @app.post("/train/pdf")
# async def train_from_pdf(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
#     """Train from PDF file upload"""
#     try:
#         # Validate file type
#         if not file.filename.endswith('.pdf'):
#             raise HTTPException(status_code=400, detail="File must be a PDF")
        
#         # Read file content
#         file_content = await file.read()
#         file_base64 = base64.b64encode(file_content).decode('utf-8')
        
#         # Parse metadata if provided
#         import json
#         parsed_metadata = json.loads(metadata) if metadata else {}
#         parsed_metadata["filename"] = file.filename
        
#         request = TrainingRequest(
#             document_type=DocumentType.PDF,
#             content=file_base64,
#             metadata=parsed_metadata
#         )
        
#         return await train_chatbot(request)
        
#     except Exception as e:
#         logger.error(f"PDF training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"PDF training failed: {str(e)}")

# @app.post("/train/qa")
# async def train_from_qa(qa_text: str = Form(...), metadata: Optional[str] = Form(None)):
#     """Train from Q&A text"""
#     try:
#         # Parse metadata if provided
#         import json
#         parsed_metadata = json.loads(metadata) if metadata else {}
        
#         request = TrainingRequest(
#             document_type=DocumentType.QA,
#             content=qa_text,
#             metadata=parsed_metadata
#         )
        
#         return await train_chatbot(request)
        
#     except Exception as e:
#         logger.error(f"Q&A training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Q&A training failed: {str(e)}")

# @app.post("/train/batch")
# async def batch_train(requests: List[TrainingRequest]):
#     """Train with multiple documents in batch"""
#     try:
#         logger.info(f"Batch training request received for {len(requests)} documents")
        
#         # Convert to list of dicts for the service
#         training_data = []
#         for req in requests:
#             training_data.append({
#                 "document_type": req.document_type.value,
#                 "content": req.content,
#                 "metadata": req.metadata
#             })
        
#         result = await training_service.batch_train(training_data)
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Batch training error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Batch training failed: {str(e)}")

# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_bot(request: ChatRequest):
#     """Chat with the trained bot"""
#     try:
#         logger.info(f"Chat request received: {request.message[:50]}...")
        
#         result = await chat_service.chat(
#             message=request.message,
#             conversation_id=request.conversation_id,
#             max_tokens=request.max_tokens,
#             temperature=request.temperature
#         )
        
#         return ChatResponse(
#             response=result["response"],
#             conversation_id=result["conversation_id"],
#             sources=result["sources"]
#         )
        
#     except Exception as e:
#         logger.error(f"Chat error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# @app.get("/chat/conversations")
# async def get_active_conversations():
#     """Get list of active conversations"""
#     try:
#         conversations = chat_service.get_active_conversations()
#         return {
#             "success": True,
#             "conversations": conversations,
#             "count": len(conversations)
#         }
#     except Exception as e:
#         logger.error(f"Error getting conversations: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

# @app.get("/chat/conversations/{conversation_id}")
# async def get_conversation_history(conversation_id: str):
#     """Get conversation history"""
#     try:
#         history = chat_service.get_conversation_history(conversation_id)
#         return {
#             "success": True,
#             "conversation_id": conversation_id,
#             "history": history,
#             "message_count": len(history)
#         }
#     except Exception as e:
#         logger.error(f"Error getting conversation history: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

# @app.delete("/chat/conversations/{conversation_id}")
# async def clear_conversation(conversation_id: str):
#     """Clear conversation history"""
#     try:
#         success = chat_service.clear_conversation(conversation_id)
#         if success:
#             return {
#                 "success": True,
#                 "message": f"Conversation {conversation_id} cleared successfully"
#             }
#         else:
#             raise HTTPException(status_code=404, detail="Conversation not found")
#     except Exception as e:
#         logger.error(f"Error clearing conversation: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

# @app.get("/training/stats")
# async def get_training_stats():
#     """Get training statistics"""
#     try:
#         stats = training_service.get_training_stats()
#         return stats
#     except Exception as e:
#         logger.error(f"Error getting training stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")

# @app.delete("/training/source/{source}")
# async def delete_documents_by_source(source: str):
#     """Delete all documents from a specific source"""
#     try:
#         result = training_service.delete_documents_by_source(source)
#         return result
#     except Exception as e:
#         logger.error(f"Error deleting documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")

# # Error handlers
# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     return {
#         "success": False,
#         "message": "Endpoint not found",
#         "detail": str(exc)
#     }

# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     return {
#         "success": False,
#         "message": "Internal server error",
#         "detail": str(exc)
#     }

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from typing import List, Optional
import base64

# Import our custom modules
from models import (
    TrainingRequest, 
    ChatRequest, 
    ChatResponse, 
    TrainingResponse, 
    HealthResponse,
    DocumentType
)
from training_service import TrainingService
from chat_service import ChatService
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Chatbot Backend",
    description="A FastAPI backend for RAG-based AI chatbot with training capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
training_service = TrainingService()
chat_service = ChatService()

# Embedding Service Class
class EmbeddingService:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self.vector_store = chat_service.vector_store
    
    def get_embeddings_for_query(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieve embeddings and metadata for a query without LLM generation
        """
        try:
            # Retrieve relevant documents from vector store
            sources = self.vector_store.similarity_search(query, top_k=top_k)
            
            # Prepare detailed embedding information
            embeddings_data = []
            for i, source in enumerate(sources):
                embedding_info = {
                    "rank": i + 1,
                    "document_id": source["id"],
                    "similarity_score": round(source["score"], 6),
                    "text_content": source["text"],
                    "embedding_vector": source.get("embedding", None),  # The actual embedding vector
                    "metadata": {
                        "source": source["metadata"].get("source", "unknown"),
                        "document_type": source["metadata"].get("document_type", "unknown"),
                        "chunk_index": source["metadata"].get("chunk_index", None),
                        "file_path": source["metadata"].get("file_path", None),
                        "created_at": source["metadata"].get("created_at", None)
                    }
                }
                embeddings_data.append(embedding_info)
            
            return {
                "query": query,
                "total_results": len(embeddings_data),
                "embeddings": embeddings_data,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }

# Initialize embedding service
embedding_service = EmbeddingService(chat_service)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="RAG AI Chatbot Backend is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # You could add more sophisticated health checks here
        # like checking database connections, etc.
        return HealthResponse(
            status="healthy",
            message="All services are operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/train", response_model=TrainingResponse)
async def train_chatbot(request: TrainingRequest):
    """Train the chatbot with new content"""
    try:
        logger.info(f"Training request received for {request.document_type}")
        
        result = await training_service.train_from_document(
            request.document_type,
            request.content,
            request.metadata
        )
        
        if result["success"]:
            return TrainingResponse(
                success=True,
                message=result["message"],
                chunks_processed=result["chunks_processed"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/train/url")
async def train_from_url(url: str = Form(...), metadata: Optional[str] = Form(None)):
    """Train from website URL"""
    try:
        # Parse metadata if provided
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        request = TrainingRequest(
            document_type=DocumentType.URL,
            content=url,
            metadata=parsed_metadata
        )
        
        return await train_chatbot(request)
        
    except Exception as e:
        logger.error(f"URL training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL training failed: {str(e)}")

@app.post("/train/pdf")
async def train_from_pdf(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
    """Train from PDF file upload"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Parse metadata if provided
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        parsed_metadata["filename"] = file.filename
        
        request = TrainingRequest(
            document_type=DocumentType.PDF,
            content=file_base64,
            metadata=parsed_metadata
        )
        
        return await train_chatbot(request)
        
    except Exception as e:
        logger.error(f"PDF training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF training failed: {str(e)}")

@app.post("/train/qa")
async def train_from_qa(qa_text: str = Form(...), metadata: Optional[str] = Form(None)):
    """Train from Q&A text"""
    try:
        # Parse metadata if provided
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        request = TrainingRequest(
            document_type=DocumentType.QA,
            content=qa_text,
            metadata=parsed_metadata
        )
        
        return await train_chatbot(request)
        
    except Exception as e:
        logger.error(f"Q&A training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Q&A training failed: {str(e)}")

@app.post("/train/batch")
async def batch_train(requests: List[TrainingRequest]):
    """Train with multiple documents in batch"""
    try:
        logger.info(f"Batch training request received for {len(requests)} documents")
        
        # Convert to list of dicts for the service
        training_data = []
        for req in requests:
            training_data.append({
                "document_type": req.document_type.value,
                "content": req.content,
                "metadata": req.metadata
            })
        
        result = await training_service.batch_train(training_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Batch training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch training failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Chat with the trained bot"""
    try:
        logger.info(f"Chat request received: {request.message[:50]}...")
        
        result = await chat_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# NEW EMBEDDING ROUTES
@app.post("/embeddings")
async def get_embeddings(query: str = Form(...), top_k: Optional[int] = Form(5)):
    """
    Get embeddings for a query without LLM generation
    
    Returns detailed embedding information including vectors, similarity scores, and metadata
    """
    try:
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise HTTPException(
                status_code=400, 
                detail="top_k must be an integer between 1 and 50"
            )
        
        logger.info(f"Embedding request received: {query[:50]}...")
        
        # Get embeddings
        result = embedding_service.get_embeddings_for_query(query, top_k)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding retrieval failed: {str(e)}")

@app.post("/embeddings/raw")
async def get_raw_embeddings(query: str = Form(...), top_k: Optional[int] = Form(5)):
    """
    Get only the raw embedding vectors without full metadata
    
    Returns a simplified response with just the embedding vectors and similarity scores
    """
    try:
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            raise HTTPException(
                status_code=400, 
                detail="top_k must be an integer between 1 and 50"
            )
        
        logger.info(f"Raw embedding request received: {query[:50]}...")
        
        # Get embeddings
        result = embedding_service.get_embeddings_for_query(query, top_k)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract only embedding vectors and basic info
        raw_embeddings = []
        for embedding_data in result["embeddings"]:
            raw_embeddings.append({
                "rank": embedding_data["rank"],
                "similarity_score": embedding_data["similarity_score"],
                "embedding_vector": embedding_data["embedding_vector"],
                "text_preview": embedding_data["text_content"][:200] + "..." if len(embedding_data["text_content"]) > 200 else embedding_data["text_content"]
            })
        
        return {
            "query": query,
            "total_results": len(raw_embeddings),
            "embeddings": raw_embeddings,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in raw embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Raw embedding retrieval failed: {str(e)}")

@app.get("/embeddings/search/{query}")
async def search_embeddings_get(query: str, top_k: int = 5):
    """
    Alternative GET endpoint for embedding search
    """
    try:
        if top_k < 1 or top_k > 50:
            raise HTTPException(
                status_code=400, 
                detail="top_k must be between 1 and 50"
            )
        
        logger.info(f"GET embedding search: {query[:50]}...")
        
        result = embedding_service.get_embeddings_for_query(query, top_k)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in GET embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding search failed: {str(e)}")

@app.get("/chat/conversations")
async def get_active_conversations():
    """Get list of active conversations"""
    try:
        conversations = chat_service.get_active_conversations()
        return {
            "success": True,
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@app.get("/chat/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        history = chat_service.get_conversation_history(conversation_id)
        return {
            "success": True,
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

@app.delete("/chat/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    try:
        success = chat_service.clear_conversation(conversation_id)
        if success:
            return {
                "success": True,
                "message": f"Conversation {conversation_id} cleared successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

@app.get("/training/stats")
async def get_training_stats():
    """Get training statistics"""
    try:
        stats = training_service.get_training_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting training stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")

@app.delete("/training/source/{source}")
async def delete_documents_by_source(source: str):
    """Delete all documents from a specific source"""
    try:
        result = training_service.delete_documents_by_source(source)
        return result
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "success": False,
        "message": "Endpoint not found",
        "detail": str(exc)
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "success": False,
        "message": "Internal server error",
        "detail": str(exc)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )