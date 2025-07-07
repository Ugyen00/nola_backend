from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from typing import List, Optional
import base64
import asyncio
import json
import os
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

# Import the enhanced crawler module
from crawl import (
    crawl_website, 
    scrape_page, 
    WebCrawler,
    CrawlConfig,
    ScrapeConfig,
    crawl_with_firecrawl,
    scrape_with_firecrawl,
    crawl_with_crawl4ai,
    scrape_with_crawl4ai,
    batch_scrape_with_firecrawl
)

# Simplified request models matching your exact API structure
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
    title="RAG AI Chatbot Backend with Enhanced Web Crawler",
    description="A FastAPI backend for RAG-based AI chatbot with training capabilities, web crawling, and Firecrawl integration with change tracking",
    version="2.0.0"
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

class SimpleCrawlerService:
    """Simplified service class to handle web crawling operations matching your API structure"""
    
    def __init__(self, training_service: TrainingService):
        self.training_service = training_service
        # Check if Firecrawl is available
        self.firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        self.use_firecrawl = bool(self.firecrawl_key)
    
    async def crawl_and_train(self, request: CrawlWebsiteRequest) -> dict:
        """Crawl website with auto method selection (Firecrawl preferred, fallback to crawl4ai)"""
        try:
            base_url = str(request.base_url)
            logger.info(f"Starting crawl for {base_url} (auto_train: {request.auto_train})")
            
            # Try Firecrawl first if available
            if self.use_firecrawl:
                try:
                    result = await crawl_with_firecrawl(
                        base_url=base_url,
                        chatbot_id=request.chatbot_id,
                        api_key=self.firecrawl_key,
                        max_pages=request.max_pages,
                        enable_change_tracking=True,
                        max_age=3600000  # 1 hour cache for performance
                    )
                    
                    if result['success']:
                        # If auto_train is False, return just the crawled data
                        if not request.auto_train:
                            # Convert to match your expected output format
                            return {
                                "success": True,
                                "message": f"Successfully crawled {result['total_pages']} pages",
                                "total_pages": result['total_pages'],
                                "successful_pages": result['total_pages'],
                                "websites_data": result['data']
                            }
                        
                        # Train with crawled data
                        return await self._train_from_crawl_data(result, request.chatbot_id, "firecrawl")
                
                except Exception as e:
                    logger.warning(f"Firecrawl failed, falling back to crawl4ai: {str(e)}")
            
            # Fallback to crawl4ai
            result = await crawl_with_crawl4ai(
                base_url=base_url,
                chatbot_id=request.chatbot_id,
                max_pages=request.max_pages,
                max_depth=request.max_depth
            )
            
            if not result['success']:
                return {
                    "success": False,
                    "message": "Failed to crawl website",
                    "crawled_pages": 0,
                    "trained_chunks": 0
                }
            
            # If auto_train is False, return just the crawled data
            if not request.auto_train:
                return {
                    "success": True,
                    "message": f"Successfully crawled {result['total_pages']} pages",
                    "total_pages": result['total_pages'],
                    "successful_pages": result['total_pages'],
                    "websites_data": result['data']
                }
            
            # Train with crawled data
            return await self._train_from_crawl_data(result, request.chatbot_id, "crawl4ai")
            
        except Exception as e:
            logger.error(f"Error in crawl_and_train: {str(e)}")
            return {
                "success": False,
                "message": f"Crawling failed: {str(e)}",
                "crawled_pages": 0,
                "trained_chunks": 0
            }
    
    async def scrape_and_train(self, request: ScrapePageRequest) -> dict:
        """Scrape single page with auto method selection"""
        try:
            url = str(request.url)
            logger.info(f"Starting scrape for {url} (auto_train: {request.auto_train})")
            
            # Try Firecrawl first if available
            if self.use_firecrawl:
                try:
                    result = await scrape_with_firecrawl(
                        url=url,
                        chatbot_id=request.chatbot_id,
                        api_key=self.firecrawl_key,
                        enable_change_tracking=True,
                        max_age=3600000  # 1 hour cache for performance
                    )
                    
                    if result['success']:
                        # If auto_train is False, return just the scraped data
                        if not request.auto_train:
                            return {
                                "success": True,
                                "message": f"Successfully scraped {url}",
                                "website_data": result['data']
                            }
                        
                        # Train with scraped data
                        return await self._train_from_scrape_data(result, url, "firecrawl")
                
                except Exception as e:
                    logger.warning(f"Firecrawl failed, falling back to crawl4ai: {str(e)}")
            
            # Fallback to crawl4ai
            result = await scrape_with_crawl4ai(
                url=url,
                chatbot_id=request.chatbot_id
            )
            
            if not result['success']:
                return {
                    "success": False,
                    "message": "Failed to scrape page",
                    "url": url,
                    "trained_chunks": 0
                }
            
            # If auto_train is False, return just the scraped data
            if not request.auto_train:
                return {
                    "success": True,
                    "message": f"Successfully scraped {url}",
                    "website_data": result['data']
                }
            
            # Train with scraped data
            return await self._train_from_scrape_data(result, url, "crawl4ai")
            
        except Exception as e:
            logger.error(f"Error in scrape_and_train: {str(e)}")
            return {
                "success": False,
                "message": f"Scraping failed: {str(e)}",
                "url": url,
                "trained_chunks": 0
            }
    
    async def _train_from_crawl_data(self, crawl_result: dict, chatbot_id: str, method: str) -> dict:
        """Train from crawl data"""
        total_chunks_trained = 0
        successful_trainings = 0
        
        for page_data in crawl_result['data']:
            try:
                # Extract content text
                if isinstance(page_data.get('content'), list):
                    content_text = "\n\n".join([chunk.get('content', '') for chunk in page_data['content']])
                else:
                    content_text = page_data.get('content', '')
                
                if not content_text.strip():
                    continue
                
                # Prepare metadata
                training_metadata = {
                    "source_type": "website_crawl",
                    "source_url": page_data.get('url', ''),
                    "title": page_data.get('title', ''),
                    "chatbot_id": chatbot_id,
                    "crawl_metadata": page_data.get('metadata', {}),
                    "method": method
                }
                
                # Add change tracking if available
                if page_data.get('change_tracking'):
                    training_metadata["change_tracking"] = page_data['change_tracking']
                
                # Train with the content
                training_result = await self.training_service.train_from_document(
                    DocumentType.TEXT,
                    content_text,
                    training_metadata
                )
                
                if training_result["success"]:
                    total_chunks_trained += training_result["chunks_processed"]
                    successful_trainings += 1
                    logger.info(f"Successfully trained from {page_data.get('url', 'unknown')}")
                else:
                    logger.error(f"Failed to train from {page_data.get('url', 'unknown')}: {training_result['message']}")
            
            except Exception as e:
                logger.error(f"Error training from page: {str(e)}")
                continue
        
        return {
            "success": True,
            "message": f"Successfully crawled {crawl_result['total_pages']} pages and trained {successful_trainings} pages",
            "crawled_pages": crawl_result['total_pages'],
            "successful_trainings": successful_trainings,
            "total_chunks_trained": total_chunks_trained,
            "chatbot_id": chatbot_id,
            "method": method
        }
    
    async def _train_from_scrape_data(self, scrape_result: dict, url: str, method: str) -> dict:
        """Train from scrape data"""
        try:
            page_data = scrape_result['data']
            
            # Extract content text
            if isinstance(page_data.get('content'), list):
                content_text = "\n\n".join([chunk.get('content', '') for chunk in page_data['content']])
            else:
                content_text = page_data.get('content', '')
            
            if not content_text.strip():
                return {
                    "success": False,
                    "message": "No content to train with",
                    "url": url,
                    "trained_chunks": 0
                }
            
            # Prepare metadata
            training_metadata = {
                "source_type": "website_scrape",
                "source_url": page_data.get('url', url),
                "title": page_data.get('title', ''),
                "chatbot_id": page_data.get('chatbot_id', ''),
                "scrape_metadata": page_data.get('metadata', {}),
                "method": method
            }
            
            # Add change tracking if available
            if page_data.get('change_tracking'):
                training_metadata["change_tracking"] = page_data['change_tracking']
            
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
                    "title": page_data.get('title', ''),
                    "chunks_processed": training_result["chunks_processed"],
                    "word_count": page_data.get('metadata', {}).get('word_count', 0),
                    "method": method
                }
            else:
                return {
                    "success": False,
                    "message": f"Scraping succeeded but training failed: {training_result['message']}",
                    "url": url,
                    "trained_chunks": 0
                }
                
        except Exception as e:
            logger.error(f"Error in training from scrape data: {str(e)}")
            return {
                "success": False,
                "message": f"Training failed: {str(e)}",
                "url": url,
                "trained_chunks": 0
            }

# Initialize services
embedding_service = EmbeddingService(chat_service)
crawler_service = SimpleCrawlerService(training_service)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="RAG AI Chatbot Backend with Enhanced Web Crawler is running"
    )

# Main crawler endpoints matching your exact API structure
@app.post("/crawler/crawl-website")
async def crawl_website_endpoint(request: CrawlWebsiteRequest):
    """
    Crawl an entire website - auto-selects best method (Firecrawl preferred, crawl4ai fallback)
    
    Input:
    {
        "chatbot_id": "test-chatbot-123",
        "base_url": "https://www.nomindbhutan.com",
        "max_pages": 10,
        "max_depth": 5,
        "auto_train": false
    }
    """
    try:
        # Validate parameters
        if request.max_pages < 1 or request.max_pages > 200:
            raise HTTPException(status_code=400, detail="max_pages must be between 1 and 200")
        
        if request.max_depth < 1 or request.max_depth > 5:
            raise HTTPException(status_code=400, detail="max_depth must be between 1 and 5")
        
        result = await crawler_service.crawl_and_train(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Website crawling error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Website crawling failed: {str(e)}")

@app.post("/crawler/scrape-page")
async def scrape_page_endpoint(request: ScrapePageRequest):
    """
    Scrape a single page - auto-selects best method (Firecrawl preferred, crawl4ai fallback)
    
    Input:
    {
        "chatbot_id": "single-page-bot-789",
        "url": "https://www.bob.bt/business-banking/loans-business/",
        "auto_train": false
    }
    """
    try:
        result = await crawler_service.scrape_and_train(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Page scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Page scraping failed: {str(e)}")

@app.post("/crawler/batch-urls")
async def batch_crawl_urls(request: BatchUrlsRequest):
    """
    Scrape multiple URLs in batch with optional Firecrawl integration
    """
    try:
        if len(request.urls) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 URLs allowed per batch")
        
        # Validate URLs
        for url in request.urls:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail=f"Invalid URL format: {url}")
        
        # Process each URL individually using the same logic as single scrape
        results = []
        successful_count = 0
        total_chunks_trained = 0
        
        for url in request.urls:
            try:
                scrape_request = ScrapePageRequest(
                    chatbot_id=request.chatbot_id,
                    url=url,
                    auto_train=request.auto_train
                )
                
                result = await crawler_service.scrape_and_train(scrape_request)
                
                if result["success"]:
                    successful_count += 1
                    if request.auto_train:
                        total_chunks_trained += result.get("chunks_processed", 0)
                
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

# Additional utility endpoints
@app.get("/crawler/status")
async def get_crawler_status():
    """Get crawler status and available methods"""
    try:
        firecrawl_available = bool(os.getenv("FIRECRAWL_API_KEY"))
        
        return {
            "success": True,
            "methods": {
                "crawl4ai": {
                    "available": True,
                    "description": "Basic web crawling using crawl4ai"
                },
                "firecrawl": {
                    "available": firecrawl_available,
                    "description": "Enhanced crawling with change tracking and caching",
                    "features": ["change_tracking", "caching", "advanced_extraction"] if firecrawl_available else []
                }
            },
            "auto_selection": True,
            "preferred_method": "firecrawl" if firecrawl_available else "crawl4ai",
            "fallback_method": "crawl4ai"
        }
    except Exception as e:
        logger.error(f"Error getting crawler status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get crawler status: {str(e)}")

# Legacy support - remove these if not needed
@app.post("/crawler/crawl-website-legacy")
async def crawl_website_legacy():
    """Legacy endpoint - redirects to main endpoint"""
    return {
        "message": "This endpoint is deprecated. Please use /crawler/crawl-website",
        "new_endpoint": "/crawler/crawl-website"
    }

@app.post("/crawler/scrape-page-legacy")  
async def scrape_page_legacy():
    """Legacy endpoint - redirects to main endpoint"""
    return {
        "message": "This endpoint is deprecated. Please use /crawler/scrape-page", 
        "new_endpoint": "/crawler/scrape-page"
    }

# Remove the complex enhanced endpoints and keep only essential ones
# The BatchUrlsRequest model and Firecrawl-specific models can be removed if not needed

# Existing training and chat endpoints (unchanged)
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

# Health and status endpoints
@app.get("/status/features")
async def get_available_features():
    """Get list of available features and their status"""
    try:
        firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
        
        return {
            "success": True,
            "features": {
                "basic_crawling": {
                    "available": True,
                    "method": "crawl4ai",
                    "description": "Basic website crawling and scraping"
                },
                "enhanced_crawling": {
                    "available": bool(firecrawl_key),
                    "method": "firecrawl",
                    "description": "Enhanced crawling with advanced features"
                },
                "change_tracking": {
                    "available": bool(firecrawl_key),
                    "method": "firecrawl",
                    "description": "Track changes between crawls"
                },
                "caching": {
                    "available": bool(firecrawl_key),
                    "method": "firecrawl",
                    "description": "Cache results for faster re-crawls"
                },
                "batch_processing": {
                    "available": True,
                    "method": "both",
                    "description": "Process multiple URLs at once"
                },
                "structured_extraction": {
                    "available": bool(firecrawl_key),
                    "method": "firecrawl",
                    "description": "Extract structured data with AI"
                }
            },
            "configuration": {
                "firecrawl_configured": bool(firecrawl_key),
                "max_pages_per_crawl": 200,
                "max_urls_per_batch": 50,
                "max_crawl_depth": 5
            }
        }
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)