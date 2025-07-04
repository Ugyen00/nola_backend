from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from typing import List, Optional
import base64
import asyncio
import json
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
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

# Import the crawler module
from crawl import crawl_website, scrape_page, WebCrawler


# Add these new models after your existing imports but before the logging config
class CrawlWebsiteRequest(BaseModel):
    chatbot_id: str
    base_url: HttpUrl
    max_pages: int = 50
    max_depth: int = 2
    auto_train: bool = True

class ScrapePageRequest(BaseModel):
    chatbot_id: str
    url: HttpUrl
    auto_train: bool = True

class BatchUrlsRequest(BaseModel):
    chatbot_id: str
    urls: List[str]
    auto_train: bool = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Chatbot Backend with Web Crawler",
    description="A FastAPI backend for RAG-based AI chatbot with training capabilities and web crawling",
    version="1.1.0"
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

class EmbeddingService:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self.vector_store = chat_service.vector_store
    
    def get_embeddings_for_query(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieve embeddings and metadata for a query - simplified version
        """
        try:
            # Retrieve relevant documents from vector store
            sources = self.vector_store.similarity_search(query, top_k=top_k)
            
            # Prepare simplified embedding information
            embeddings_data = []
            for i, source in enumerate(sources):
                embedding_info = {
                    "rank": i + 1,
                    "document_id": source["id"],
                    "similarity_score": round(source["score"], 5),
                    "content": source["text"],  # This should now contain your content
                    "source_id": source["metadata"].get("source_id", ""),
                    "source_type": source["metadata"].get("source_type", ""),
                    "chatbot_id": source["metadata"].get("chatbot_id", "")
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

class CrawlerService:
    """Service class to handle web crawling operations"""
    
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
        self.crawler = WebCrawler()
    
    async def crawl_and_train(self, base_url: str, chatbot_id: str, max_pages: int = 50, max_depth: int = 2, auto_train: bool = True) -> dict:
        """Crawl a website and optionally train the chatbot with the content"""
        try:
            logger.info(f"Starting crawl and train for {base_url}")
            
            # Crawl the website
            websites_data = await crawl_website(base_url, chatbot_id, max_pages, max_depth)
            
            if not websites_data:
                return {
                    "success": False,
                    "message": "No content was successfully crawled from the website",
                    "crawled_pages": 0,
                    "trained_chunks": 0,
                    "websites_data": []
                }
            
            # If auto_train is False, return just the crawled data
            if not auto_train:
                return {
                    "success": True,
                    "message": f"Successfully crawled {len(websites_data)} pages",
                    "total_pages": len(websites_data),
                    "successful_pages": len(websites_data),
                    "websites_data": [website.to_dict() for website in websites_data]
                }
            
            # Train the chatbot with each crawled page
            total_chunks_trained = 0
            successful_trainings = 0
            
            for website_data in websites_data:
                try:
                    # Convert website data to training format
                    content_text = "\n\n".join([chunk.content for chunk in website_data.content])
                    
                    # Prepare metadata for training
                    training_metadata = {
                        "source_type": "website_crawl",
                        "source_url": website_data.url,
                        "title": website_data.title,
                        "chatbot_id": chatbot_id,
                        "crawl_metadata": website_data.metadata
                    }
                    
                    # Train with the content
                    training_result = await self.training_service.train_from_document(
                        DocumentType.TEXT,
                        content_text,
                        training_metadata
                    )
                    
                    if training_result["success"]:
                        total_chunks_trained += training_result["chunks_processed"]
                        successful_trainings += 1
                        logger.info(f"Successfully trained from {website_data.url}")
                    else:
                        logger.error(f"Failed to train from {website_data.url}: {training_result['message']}")
                
                except Exception as e:
                    logger.error(f"Error training from {website_data.url}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "message": f"Successfully crawled {len(websites_data)} pages and trained {successful_trainings} pages",
                "crawled_pages": len(websites_data),
                "successful_trainings": successful_trainings,
                "total_chunks_trained": total_chunks_trained,
                "base_url": base_url,
                "chatbot_id": chatbot_id
            }
            
        except Exception as e:
            logger.error(f"Error in crawl_and_train: {str(e)}")
            return {
                "success": False,
                "message": f"Crawling and training failed: {str(e)}",
                "crawled_pages": 0,
                "trained_chunks": 0
            }
    
    async def scrape_and_train(self, url: str, chatbot_id: str, auto_train: bool = True) -> dict:
        """Scrape a single page and optionally train the chatbot with the content"""
        try:
            logger.info(f"Starting scrape and train for {url}")
            
            # Scrape the page
            website_data = await scrape_page(url, chatbot_id)
            
            if not website_data:
                return {
                    "success": False,
                    "message": "Failed to scrape content from the URL",
                    "url": url,
                    "trained_chunks": 0
                }
            
            # If auto_train is False, return just the scraped data
            if not auto_train:
                return {
                    "success": True,
                    "message": f"Successfully scraped {url}",
                    "website_data": website_data.to_dict()
                }
            
            # Convert to training format
            content_text = "\n\n".join([chunk.content for chunk in website_data.content])
            
            # Prepare metadata
            training_metadata = {
                "source_type": "website_scrape",
                "source_url": website_data.url,
                "title": website_data.title,
                "chatbot_id": chatbot_id,
                "scrape_metadata": website_data.metadata
            }
            
            # Train with the content
            training_result = await self.training_service.train_from_document(
                DocumentType.TEXT,
                content_text,
                training_metadata
            )
            
            if training_result["success"]:
                return {
                    "success": True,
                    "message": f"Successfully scraped and trained from {url}",
                    "url": url,
                    "title": website_data.title,
                    "chunks_processed": training_result["chunks_processed"],
                    "word_count": website_data.metadata.get("word_count", 0)
                }
            else:
                return {
                    "success": False,
                    "message": f"Scraping succeeded but training failed: {training_result['message']}",
                    "url": url,
                    "trained_chunks": 0
                }
                
        except Exception as e:
            logger.error(f"Error in scrape_and_train: {str(e)}")
            return {
                "success": False,
                "message": f"Scraping and training failed: {str(e)}",
                "url": url,
                "trained_chunks": 0
            }
        
# Initialize services
embedding_service = EmbeddingService(chat_service)
crawler_service = CrawlerService(training_service)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="RAG AI Chatbot Backend with Web Crawler is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            message="All services are operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/crawler/crawl-website")
async def crawl_website_endpoint(request: CrawlWebsiteRequest):
    """
    Crawl an entire website and optionally train the chatbot with the content (JSON input)
    """
    try:
        # Validate parameters
        if request.max_pages < 1 or request.max_pages > 200:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 200")
        
        if request.max_depth < 1 or request.max_depth > 5:
            raise HTTPException(status_code=400, detail="max_depth must be between 1 and 5")
        
        base_url = str(request.base_url)
        
        # Use the updated crawl_and_train method that handles both auto_train cases
        result = await crawler_service.crawl_and_train(
            base_url, 
            request.chatbot_id, 
            request.max_pages, 
            request.max_depth,
            request.auto_train
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Website crawling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Website crawling failed: {str(e)}")

@app.post("/crawler/scrape-page")
async def scrape_page_endpoint(request: ScrapePageRequest):
    """
    Scrape a single page and optionally train the chatbot with the content (JSON input)
    """
    try:
        url = str(request.url)
        
        # Use the updated scrape_and_train method that handles both auto_train cases
        result = await crawler_service.scrape_and_train(url, request.chatbot_id, request.auto_train)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Page scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Page scraping failed: {str(e)}")

@app.post("/crawler/batch-urls")
async def batch_crawl_urls(request: BatchUrlsRequest):
    """
    Scrape multiple URLs in batch and optionally train the chatbot (JSON input)
    """
    try:
        if len(request.urls) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 URLs allowed per batch")
        
        # Validate URLs
        for url in request.urls:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
        
        # Process each URL
        results = []
        successful_count = 0
        total_chunks_trained = 0
        
        for url in request.urls:
            try:
                if request.auto_train:
                    result = await crawler_service.scrape_and_train(url, request.chatbot_id, True)
                    if result["success"]:
                        successful_count += 1
                        total_chunks_trained += result.get("chunks_processed", 0)
                else:
                    result = await crawler_service.scrape_and_train(url, request.chatbot_id, False)
                    if result["success"]:
                        successful_count += 1
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                results.append({
                    "success": False,
                    "url": url,
                    "message": f"Error: {str(e)}"
                })
        
        return {
            "success": True,
            "message": f"Batch processing completed: {successful_count}/{len(request.urls)} URLs successful",
            "total_urls": len(request.urls),
            "successful_urls": successful_count,
            "total_chunks_trained": total_chunks_trained if request.auto_train else None,
            "results": results,
            "chatbot_id": request.chatbot_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch URL processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch URL processing failed: {str(e)}")

@app.get("/crawler/status")
async def crawler_status():
    """Get crawler service status and capabilities"""
    try:
        return {
            "status": "operational",
            "capabilities": {
                "website_crawling": True,
                "single_page_scraping": True,
                "batch_processing": True,
                "auto_training": True,
                "sitemap_discovery": True,
                "link_following": True
            },
            "limits": {
                "max_pages_per_crawl": 200,
                "max_depth": 5,
                "max_batch_urls": 50,
                "default_chunk_size": 1000
            },
            "supported_content_types": [
                "text/html",
                "application/xhtml+xml"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting crawler status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get crawler status: {str(e)}")

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
    """Train from website URL (legacy endpoint - consider using /crawler/scrape-page instead)"""
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

@app.post("/train/file")
async def train_from_file(
    file: UploadFile = File(...),
    file_type: str = Form(...),  # "pdf", "docx", "xlsx", or "txt"
    metadata: Optional[str] = Form(None)
):
    """Train from uploaded file"""
    try:
        supported_types = ["pdf", "docx", "xlsx", "txt"]
        if file_type not in supported_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        # Read and encode file
        file_content = await file.read()
        file_base64 = base64.b64encode(file_content).decode("utf-8")
        
        # Parse metadata
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        parsed_metadata["filename"] = file.filename
        
        request = TrainingRequest(
            document_type=file_type,
            content=file_base64,
            metadata=parsed_metadata
        )
        
        return await train_chatbot(request)

    except Exception as e:
        logger.error(f"{file_type.upper()} training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{file_type.upper()} training failed: {str(e)}")

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

@app.post("/train/text")
async def train_from_text(text_content: str = Form(...), title: Optional[str] = Form(None), metadata: Optional[str] = Form(None)):
    """Train from raw text content"""
    try:
        # Validate input
        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        # Parse metadata if provided
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        # Add title to metadata if provided
        if title:
            parsed_metadata["title"] = title
        
        request = TrainingRequest(
            document_type=DocumentType.TEXT,
            content=text_content,
            metadata=parsed_metadata
        )
        
        return await train_chatbot(request)
        
    except Exception as e:
        logger.error(f"Text training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text training failed: {str(e)}")

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

@app.post("/embeddings")
async def get_embeddings(query: str = Form(...), top_k: Optional[int] = Form(5)):
    """
    Get embeddings for a query - simplified response
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
                "document_id": embedding_data["document_id"],
                "embedding_vector": embedding_data.get("embedding_vector", []),
                "text_preview": embedding_data["content"][:200] + "..." if len(embedding_data["content"]) > 200 else embedding_data["content"],
                "source_info": {
                    "source_id": embedding_data["source_id"],
                    "source_type": embedding_data["source_type"],
                    "chatbot_id": embedding_data["chatbot_id"]
                }
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
    GET endpoint for embedding search
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

# Endpoint to get unique values for filtering
@app.get("/embeddings/metadata/values")
async def get_metadata_values():
    """
    Get unique values for metadata fields to help with filtering
    """
    try:
        # Note: This would require additional implementation in your vector store
        # For now, returning a placeholder response
        return {
            "message": "This endpoint would return unique values for chatbot_id, source_type, and source_id",
            "note": "Implementation depends on your vector store's capability to query metadata",
            "status": "not_implemented"
        }
        
    except Exception as e:
        logger.error(f"Error in metadata values endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata values: {str(e)}")

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

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)