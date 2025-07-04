"""
Web Crawler Module
A standalone web crawler that can be integrated into any Python project
"""

import asyncio
import xml.etree.ElementTree as ET
import requests
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from bs4 import BeautifulSoup
import hashlib
from datetime import datetime
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChunkData:
    """Data class for content chunks"""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }


@dataclass
class WebsiteData:
    """Data class for scraped website data"""
    chatbot_id: str
    title: str
    url: str
    content: List[ChunkData]
    metadata: Dict[str, Any]
    trained: bool = False
    deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chatbot_id': self.chatbot_id,
            'title': self.title,
            'url': self.url,
            'content': [chunk.to_dict() for chunk in self.content],
            'metadata': self.metadata,
            'trained': self.trained,
            'deleted': self.deleted
        }


class TextChunker:
    """Handles text chunking for better content processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, url: str) -> List[ChunkData]:
        """Chunk text into smaller pieces with metadata"""
        chunks = []
        
        if not text or not text.strip():
            return chunks
        
        # Split by paragraphs first, then by tokens if needed
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if self.count_tokens(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(current_chunk, url, url_hash, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunk = self._create_chunk(current_chunk, url, url_hash, chunk_index)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, url: str, url_hash: str, chunk_index: int) -> ChunkData:
        """Create a chunk with metadata"""
        chunk_id = f"{url_hash}_chunk_{chunk_index}"
        
        metadata = {
            'url': url,
            'url_hash': url_hash,
            'chunk_index': chunk_index,
            'chunk_token_count': self.count_tokens(content),
            'chunk_char_count': len(content),
            'created_at': datetime.now().isoformat()
        }
        
        return ChunkData(
            id=chunk_id,
            content=content.strip(),
            metadata=metadata
        )


class URLDiscoverer:
    """Discovers URLs from sitemaps and page crawling"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
    
    def get_sitemap_urls(self) -> List[str]:
        """Extract URLs from sitemap.xml"""
        sitemap_urls = []
        
        # Common sitemap locations
        sitemap_locations = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml", 
            f"{self.base_url}/sitemaps.xml"
        ]
        
        for sitemap_url in sitemap_locations:
            try:
                print(f"🗺️  Checking sitemap: {sitemap_url}")
                response = requests.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    urls = self._parse_sitemap(response.content)
                    sitemap_urls.extend(urls)
                    print(f"✓ Found {len(urls)} URLs from sitemap")
                    break
            except Exception as e:
                print(f"✗ Failed to fetch sitemap: {str(e)}")
                continue
        
        return list(set(sitemap_urls))
    
    def _parse_sitemap(self, content: bytes) -> List[str]:
        """Parse XML sitemap content"""
        urls = []
        try:
            root = ET.fromstring(content)
            
            # Handle sitemap index
            if 'sitemapindex' in root.tag:
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                    loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        try:
                            sub_response = requests.get(loc.text, timeout=10)
                            if sub_response.status_code == 200:
                                urls.extend(self._parse_sitemap(sub_response.content))
                        except Exception:
                            continue
            
            # Handle regular sitemap
            else:
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        urls.append(loc.text)
        
        except ET.ParseError:
            pass
        
        return urls
    
    def extract_links_from_html(self, html_content: str, current_url: str) -> List[str]:
        """Extract internal links from HTML"""
        links = []
        
        if not html_content:
            return links
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href or href.startswith('#'):
                    continue
                
                # Convert to absolute URL
                full_url = urljoin(current_url, href)
                parsed_url = urlparse(full_url)
                
                # Only internal links
                if parsed_url.netloc == self.domain:
                    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                    if parsed_url.query:
                        clean_url += f"?{parsed_url.query}"
                    
                    # Filter out unwanted file types
                    if not any(clean_url.lower().endswith(ext) for ext in 
                              ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx']):
                        links.append(clean_url)
        
        except Exception as e:
            print(f"Error extracting links: {str(e)}")
        
        return links


class WebCrawler:
    """Main web crawler class"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
    
    async def scrape_single_page(self, url: str, chatbot_id: str) -> Optional[WebsiteData]:
        """Scrape a single page and return formatted data"""
        try:
            print(f"🔍 Scraping: {url}")
            
            async with AsyncWebCrawler() as crawler:
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=5,
                    page_timeout=30000
                )
                
                result = await crawler.arun(url=url, config=config)
                
                if result.success and result.markdown:
                    # Create chunks from content
                    chunks = self.chunker.chunk_text(result.markdown, url)
                    
                    # Format data
                    website_data = WebsiteData(
                        chatbot_id=chatbot_id,
                        title=result.metadata.get('title', 'No Title'),
                        url=url,
                        content=chunks,
                        metadata={
                            'method': 'scrape',
                            'scraped_at': datetime.now().isoformat(),
                            'word_count': len(result.markdown.split()) if result.markdown else 0,
                            'description': result.metadata.get('description', ''),
                            'keywords': result.metadata.get('keywords', '')
                        }
                    )
                    
                    print(f"✓ Successfully scraped: {len(chunks)} chunks")
                    return website_data
                else:
                    print(f"✗ Failed to scrape page")
                    return None
                    
        except Exception as e:
            print(f"✗ Error scraping {url}: {str(e)}")
            return None
    
    async def crawl_website(self, base_url: str, chatbot_id: str, 
                           max_pages: int = 50, max_depth: int = 2) -> List[WebsiteData]:
        """Crawl entire website and return formatted data"""
        print(f"🚀 Starting website crawl: {base_url}")
        
        # Discover URLs
        discoverer = URLDiscoverer(base_url)
        discovered_urls = set()
        
        # Get sitemap URLs
        sitemap_urls = discoverer.get_sitemap_urls()
        if sitemap_urls:
            discovered_urls.update(sitemap_urls)
            print(f"📋 Found {len(sitemap_urls)} URLs from sitemap")
        
        # Always include homepage
        discovered_urls.add(base_url)
        discovered_urls.add(f"{base_url}/")
        
        # Discover more URLs by following links
        await self._discover_urls_by_crawling(discoverer, discovered_urls, max_depth, max_pages)
        
        # Limit to max_pages
        urls_to_crawl = list(discovered_urls)[:max_pages]
        print(f"🎯 Will crawl {len(urls_to_crawl)} URLs")
        
        # Crawl all discovered URLs
        websites_data = []
        successful_count = 0
        
        async with AsyncWebCrawler() as crawler:
            for i, url in enumerate(urls_to_crawl, 1):
                try:
                    print(f"[{i}/{len(urls_to_crawl)}] Crawling: {url}")
                    
                    config = CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=5,
                        page_timeout=30000
                    )
                    
                    result = await crawler.arun(url=url, config=config)
                    
                    if result.success and result.markdown:
                        # Create chunks
                        chunks = self.chunker.chunk_text(result.markdown, url)
                        
                        # Format data
                        website_data = WebsiteData(
                            chatbot_id=chatbot_id,
                            title=result.metadata.get('title', 'No Title'),
                            url=url,
                            content=chunks,
                            metadata={
                                'method': 'crawl',
                                'base_url': base_url,
                                'crawled_at': datetime.now().isoformat(),
                                'word_count': len(result.markdown.split()) if result.markdown else 0,
                                'description': result.metadata.get('description', ''),
                                'keywords': result.metadata.get('keywords', '')
                            }
                        )
                        
                        websites_data.append(website_data)
                        successful_count += 1
                        print(f"✓ Success: {len(chunks)} chunks")
                    else:
                        print(f"✗ Failed to crawl")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                
                # Respectful delay
                await asyncio.sleep(1)
        
        print(f"🎉 Crawl complete: {successful_count}/{len(urls_to_crawl)} successful")
        return websites_data
    
    async def _discover_urls_by_crawling(self, discoverer: URLDiscoverer, 
                                       discovered_urls: set, max_depth: int, max_pages: int):
        """Discover URLs by following links"""
        crawled_urls = set()
        urls_to_explore = list(discovered_urls)[:10]  # Start with first 10
        depth = 0
        
        async with AsyncWebCrawler() as crawler:
            while (urls_to_explore and depth < max_depth and 
                   len(discovered_urls) < max_pages):
                
                print(f"🌐 Discovery depth {depth + 1}: exploring {len(urls_to_explore)} URLs")
                new_urls = set()
                
                for url in urls_to_explore:
                    if url in crawled_urls or len(discovered_urls) >= max_pages:
                        continue
                    
                    try:
                        config = CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS,
                            word_count_threshold=1,
                            page_timeout=20000
                        )
                        
                        result = await crawler.arun(url=url, config=config)
                        crawled_urls.add(url)
                        
                        if result.success and result.html:
                            page_links = discoverer.extract_links_from_html(result.html, url)
                            
                            for link in page_links:
                                if (link not in discovered_urls and 
                                    len(discovered_urls) < max_pages):
                                    discovered_urls.add(link)
                                    new_urls.add(link)
                        
                        await asyncio.sleep(0.5)  # Small delay
                    
                    except Exception:
                        continue
                
                urls_to_explore = list(new_urls)[:15]  # Limit for next depth
                depth += 1


# Convenience functions for easy usage
async def crawl_website(base_url: str, chatbot_id: str, max_pages: int = 50, max_depth: int = 2) -> List[WebsiteData]:
    """Convenience function to crawl a website"""
    crawler = WebCrawler()
    return await crawler.crawl_website(base_url, chatbot_id, max_pages, max_depth)


async def scrape_page(url: str, chatbot_id: str) -> Optional[WebsiteData]:
    """Convenience function to scrape a single page"""
    crawler = WebCrawler()
    return await crawler.scrape_single_page(url, chatbot_id)