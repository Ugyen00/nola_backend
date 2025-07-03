import asyncio
from crawl4ai import AsyncWebCrawler
import PyPDF2
from typing import List, Dict, Any
import io
import base64
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    async def process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process website URL and extract text content"""
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(url=url)
                
                if result.success:
                    # Extract text content
                    text_content = result.markdown or result.cleaned_html or ""
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_text(text_content)
                    
                    # Create document objects
                    documents = []
                    for i, chunk in enumerate(chunks):
                        doc = {
                            "text": chunk,
                            "metadata": {
                                "source": url,
                                "document_type": "url",
                                "chunk_index": i,
                                "title": result.title or "",
                                "url": url
                            }
                        }
                        documents.append(doc)
                    
                    logger.info(f"Processed URL {url}: {len(documents)} chunks")
                    return documents
                else:
                    logger.error(f"Failed to crawl URL {url}: {result.error_message}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return []
    
    def process_pdf(self, pdf_base64: str) -> List[Dict[str, Any]]:
        """Process PDF file from base64 encoded content"""
        try:
            # Decode base64
            pdf_bytes = base64.b64decode(pdf_base64)
            pdf_file = io.BytesIO(pdf_bytes)
            
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                text_content += page.extract_text() + "\n"
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        "source": "uploaded_pdf",
                        "document_type": "pdf",
                        "chunk_index": i,
                        "total_pages": len(pdf_reader.pages),
                        "page_range": f"chunk_{i}"
                    }
                }
                documents.append(doc)
            
            logger.info(f"Processed PDF: {len(documents)} chunks from {len(pdf_reader.pages)} pages")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return []
    
    def process_qa_text(self, qa_content: str) -> List[Dict[str, Any]]:
        """Process Q&A text content"""
        try:
            # Split Q&A content into chunks
            chunks = self.text_splitter.split_text(qa_content)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        "source": "qa_text",
                        "document_type": "qa",
                        "chunk_index": i,
                        "content_type": "question_answer"
                    }
                }
                documents.append(doc)
            
            logger.info(f"Processed Q&A text: {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing Q&A text: {str(e)}")
            return []
    
    async def process_document(self, document_type: str, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process document based on type"""
        if metadata is None:
            metadata = {}
        
        try:
            if document_type == "url":
                documents = await self.process_url(content)
            elif document_type == "pdf":
                documents = self.process_pdf(content)
            elif document_type == "qa":
                documents = self.process_qa_text(content)
            else:
                logger.error(f"Unknown document type: {document_type}")
                return []
            
            # Add additional metadata to all documents
            for doc in documents:
                doc["metadata"].update(metadata)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return []
