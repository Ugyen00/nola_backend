import asyncio
from crawl4ai import AsyncWebCrawler
import PyPDF2
from typing import List, Dict, Any
import io
import base64
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import config
import docx2txt
import pandas as pd

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
    
    def process_file(self, file_base64: str, file_type: str) -> List[Dict[str, Any]]:
        """Process various file types from base64 encoded content"""
        try:
            # Decode base64
            file_bytes = base64.b64decode(file_base64)
            file_stream = io.BytesIO(file_bytes)
            documents = []

            if file_type == "pdf":
                pdf_reader = PyPDF2.PdfReader(file_stream)
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += page.extract_text() + "\n"
                chunks = self.text_splitter.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": "uploaded_pdf",
                            "document_type": "pdf",
                            "chunk_index": i,
                            "total_pages": len(pdf_reader.pages),
                            "page_range": f"chunk_{i}"
                        }
                    })

            elif file_type == "docx":
                text_content = docx2txt.process(file_stream)
                chunks = self.text_splitter.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": "uploaded_docx",
                            "document_type": "docx",
                            "chunk_index": i
                        }
                    })

            elif file_type == "xlsx":
                excel_data = pd.read_excel(file_stream, sheet_name=None)
                combined_text = ""
                for sheet_name, df in excel_data.items():
                    combined_text += f"Sheet: {sheet_name}\n{df.to_string(index=False)}\n"
                chunks = self.text_splitter.split_text(combined_text)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": "uploaded_xlsx",
                            "document_type": "xlsx",
                            "chunk_index": i
                        }
                    })

            elif file_type == "txt":
                text_content = file_stream.read().decode("utf-8")
                chunks = self.text_splitter.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": "uploaded_txt",
                            "document_type": "txt",
                            "chunk_index": i
                        }
                    })

            else:
                raise ValueError("Unsupported file type")

            logger.info(f"Processed {file_type.upper()}: {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing {file_type}: {str(e)}")
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
    
    def process_text(self, text_content: str, title: str = None) -> List[Dict[str, Any]]:
        """Process raw text content provided by user"""
        try:
            # Validate input
            if not text_content or not text_content.strip():
                logger.warning("Empty text content provided")
                return []
            
            # Split text content into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            # Create document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        "source": "user_text",
                        "document_type": "text",
                        "chunk_index": i,
                        "title": title or "User Provided Text",
                        "content_length": len(text_content),
                        "total_chunks": len(chunks)
                    }
                }
                documents.append(doc)
            
            logger.info(f"Processed text content: {len(documents)} chunks from {len(text_content)} characters")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text content: {str(e)}")
            return []
    
    async def process_document(self, document_type: str, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process document based on type"""
        if metadata is None:
            metadata = {}
        
        try:
            if document_type == "url":
                documents = await self.process_url(content)
            elif document_type in ["pdf", "docx", "xlsx", "txt"]:
                documents = self.process_file(content, document_type)
            elif document_type == "qa":
                documents = self.process_qa_text(content)
            elif document_type == "text":
                # Extract title from metadata if provided
                title = metadata.get("title", None)
                documents = self.process_text(content, title)
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