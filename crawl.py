"""
Web Crawler Module
A standalone web crawler that can be integrated into any Python project
Enhanced with Firecrawl capabilities and change tracking
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
import json
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# Firecrawl imports
try:
    from firecrawl import FirecrawlApp, ScrapeOptions
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    print("Warning: Firecrawl not available. Some features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
class ChangeTrackingData:
    """Data class for change tracking information"""
    previous_scrape_at: Optional[str] = None
    change_status: str = "new"  # new, same, changed, removed
    visibility: str = "visible"  # visible, hidden
    diff_text: Optional[str] = None
    diff_json: Optional[Dict] = None
    json_changes: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'previous_scrape_at': self.previous_scrape_at,
            'change_status': self.change_status,
            'visibility': self.visibility,
            'diff_text': self.diff_text,
            'diff_json': self.diff_json,
            'json_changes': self.json_changes
        }


@dataclass
class WebsiteData:
    """Data class for scraped website data"""
    chatbot_id: str
    title: str
    url: str
    content: List[ChunkData]
    metadata: Dict[str, Any]
    change_tracking: Optional[ChangeTrackingData] = None
    trained: bool = False
    deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'chatbot_id': self.chatbot_id,
            'title': self.title,
            'url': self.url,
            'content': [chunk.to_dict() for chunk in self.content],
            'metadata': self.metadata,
            'trained': self.trained,
            'deleted': self.deleted
        }
        if self.change_tracking:
            result['change_tracking'] = self.change_tracking.to_dict()
        return result


@dataclass
class CrawlConfig:
    """Configuration for crawl operations"""
    method: str = "auto"  # "firecrawl", "crawl4ai", "auto"
    max_pages: int = 50
    max_depth: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_change_tracking: bool = True
    max_age: Optional[int] = 3600000  # 1 hour cache
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    formats: List[str] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['markdown', 'changeTracking'] if self.enable_change_tracking else ['markdown']


@dataclass
class ScrapeConfig:
    """Configuration for scrape operations"""
    method: str = "auto"  # "firecrawl", "crawl4ai", "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_change_tracking: bool = True
    max_age: Optional[int] = 3600000  # 1 hour cache
    only_main_content: bool = False
    formats: List[str] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['markdown', 'changeTracking'] if self.enable_change_tracking else ['markdown']


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
                logger.info(f"🗺️  Checking sitemap: {sitemap_url}")
                response = requests.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    urls = self._parse_sitemap(response.content)
                    sitemap_urls.extend(urls)
                    logger.info(f"✓ Found {len(urls)} URLs from sitemap")
                    break
            except Exception as e:
                logger.warning(f"✗ Failed to fetch sitemap: {str(e)}")
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
            logger.error(f"Error extracting links: {str(e)}")
        
        return links


class FirecrawlIntegration:
    """Integration with Firecrawl API for enhanced scraping"""
    
    def __init__(self, api_key: str):
        if not FIRECRAWL_AVAILABLE:
            raise ImportError("Firecrawl is not available. Please install it: pip install firecrawl-py")
        
        self.app = FirecrawlApp(api_key=api_key)
        self.api_key = api_key
    
    def scrape_page(self, url: str, config: ScrapeConfig) -> Optional[Dict]:
        """Scrape a single page using Firecrawl"""
        try:
            logger.info(f"🔥 Firecrawl scraping: {url}")
            
            scrape_params = {
                'formats': config.formats,
                'onlyMainContent': config.only_main_content,
                'timeout': 120000
            }
            
            if config.max_age is not None:
                scrape_params['maxAge'] = config.max_age
            
            if config.enable_change_tracking:
                scrape_params['changeTrackingOptions'] = {
                    'modes': ['git-diff', 'json'],
                    'tag': 'default'
                }
            
            result = self.app.scrape_url(url, **scrape_params)
            
            if result.success:
                return {
                    'success': True,
                    'markdown': getattr(result, 'markdown', ''),
                    'html': getattr(result, 'html', ''),
                    'metadata': result.metadata,
                    'change_tracking': getattr(result, 'changeTracking', None)
                }
            else:
                return {'success': False, 'error': 'Scrape failed'}
                
        except Exception as e:
            logger.error(f"Firecrawl scrape error for {url}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def crawl_website(self, url: str, config: CrawlConfig) -> Dict:
        """Crawl website using Firecrawl"""
        try:
            logger.info(f"🔥 Firecrawl crawling: {url}")
            
            scrape_options = ScrapeOptions(
                formats=config.formats,
                maxAge=config.max_age
            )
            
            crawl_params = {
                'limit': config.max_pages,
                'scrape_options': scrape_options
            }
            
            if config.include_paths:
                crawl_params['includePaths'] = config.include_paths
            if config.exclude_paths:
                crawl_params['excludePaths'] = config.exclude_paths
            
            result = self.app.crawl_url(url, **crawl_params)
            
            if result.success:
                return {
                    'success': True,
                    'status': result.status,
                    'total': result.total,
                    'completed': result.completed,
                    'credits_used': result.creditsUsed,
                    'data': result.data
                }
            else:
                return {'success': False, 'error': 'Crawl failed'}
                
        except Exception as e:
            logger.error(f"Firecrawl crawl error for {url}: {str(e)}")
            return {'success': False, 'error': str(e)}


class WebCrawler:
    """Main web crawler class with Firecrawl and crawl4ai support"""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl = None
        if firecrawl_api_key and FIRECRAWL_AVAILABLE:
            try:
                self.firecrawl = FirecrawlIntegration(firecrawl_api_key)
                logger.info("✓ Firecrawl integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl: {str(e)}")
        
        if not self.firecrawl:
            logger.info("Using crawl4ai for scraping")
    
    def _determine_method(self, method: str) -> str:
        """Determine which scraping method to use"""
        if method == "auto":
            return "firecrawl" if self.firecrawl else "crawl4ai"
        elif method == "firecrawl" and not self.firecrawl:
            logger.warning("Firecrawl requested but not available, falling back to crawl4ai")
            return "crawl4ai"
        return method
    
    def _parse_change_tracking(self, change_tracking_data) -> Optional[ChangeTrackingData]:
        """Parse change tracking data from Firecrawl response"""
        if not change_tracking_data:
            return None
        
        try:
            ct_data = ChangeTrackingData(
                previous_scrape_at=getattr(change_tracking_data, 'previousScrapeAt', None),
                change_status=getattr(change_tracking_data, 'changeStatus', 'new'),
                visibility=getattr(change_tracking_data, 'visibility', 'visible')
            )
            
            # Add diff data if available
            if hasattr(change_tracking_data, 'diff') and change_tracking_data.diff:
                ct_data.diff_text = getattr(change_tracking_data.diff, 'text', None)
                ct_data.diff_json = getattr(change_tracking_data.diff, 'json', None)
            
            # Add JSON changes if available
            if hasattr(change_tracking_data, 'json') and change_tracking_data.json:
                ct_data.json_changes = change_tracking_data.json
            
            return ct_data
            
        except Exception as e:
            logger.error(f"Error parsing change tracking data: {str(e)}")
            return None
    
    async def scrape_single_page(self, url: str, chatbot_id: str, 
                               config: Optional[ScrapeConfig] = None) -> Optional[WebsiteData]:
        """Scrape a single page and return formatted data"""
        if config is None:
            config = ScrapeConfig()
        
        method = self._determine_method(config.method)
        chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        
        try:
            if method == "firecrawl" and self.firecrawl:
                # Use Firecrawl
                result = self.firecrawl.scrape_page(url, config)
                
                if result and result.get('success'):
                    markdown_content = result.get('markdown', '')
                    chunks = chunker.chunk_text(markdown_content, url)
                    
                    # Parse change tracking
                    change_tracking = None
                    if config.enable_change_tracking and result.get('change_tracking'):
                        change_tracking = self._parse_change_tracking(result['change_tracking'])
                    
                    website_data = WebsiteData(
                        chatbot_id=chatbot_id,
                        title=result['metadata'].get('title', 'No Title'),
                        url=url,
                        content=chunks,
                        change_tracking=change_tracking,
                        metadata={
                            'method': 'firecrawl',
                            'scraped_at': datetime.now().isoformat(),
                            'word_count': len(markdown_content.split()) if markdown_content else 0,
                            'description': result['metadata'].get('description', ''),
                            'keywords': result['metadata'].get('keywords', ''),
                            'status_code': result['metadata'].get('statusCode', 0)
                        }
                    )
                    
                    logger.info(f"✓ Firecrawl scraped: {len(chunks)} chunks, status: {change_tracking.change_status if change_tracking else 'no tracking'}")
                    return website_data
                else:
                    logger.error(f"✗ Firecrawl failed: {result.get('error', 'Unknown error')}")
                    # Fallback to crawl4ai
                    method = "crawl4ai"
            
            if method == "crawl4ai":
                # Use crawl4ai
                logger.info(f"🔍 Crawl4ai scraping: {url}")
                
                async with AsyncWebCrawler() as crawler:
                    crawler_config = CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=5,
                        page_timeout=30000
                    )
                    
                    result = await crawler.arun(url=url, config=crawler_config)
                    
                    if result.success and result.markdown:
                        chunks = chunker.chunk_text(result.markdown, url)
                        
                        website_data = WebsiteData(
                            chatbot_id=chatbot_id,
                            title=result.metadata.get('title', 'No Title'),
                            url=url,
                            content=chunks,
                            metadata={
                                'method': 'crawl4ai',
                                'scraped_at': datetime.now().isoformat(),
                                'word_count': len(result.markdown.split()) if result.markdown else 0,
                                'description': result.metadata.get('description', ''),
                                'keywords': result.metadata.get('keywords', '')
                            }
                        )
                        
                        logger.info(f"✓ Crawl4ai scraped: {len(chunks)} chunks")
                        return website_data
                    else:
                        logger.error(f"✗ Crawl4ai failed to scrape page")
                        return None
                        
        except Exception as e:
            logger.error(f"✗ Error scraping {url}: {str(e)}")
            return None
    
    async def crawl_website(self, base_url: str, chatbot_id: str, 
                           config: Optional[CrawlConfig] = None) -> List[WebsiteData]:
        """Crawl entire website and return formatted data"""
        if config is None:
            config = CrawlConfig()
        
        method = self._determine_method(config.method)
        chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        
        logger.info(f"🚀 Starting website crawl: {base_url} (method: {method})")
        
        websites_data = []
        
        try:
            if method == "firecrawl" and self.firecrawl:
                # Use Firecrawl for crawling
                result = self.firecrawl.crawl_website(base_url, config)
                
                if result and result.get('success'):
                    logger.info(f"🔥 Firecrawl crawled {result.get('completed', 0)} pages")
                    
                    for page in result.get('data', []):
                        try:
                            markdown_content = getattr(page, 'markdown', '')
                            chunks = chunker.chunk_text(markdown_content, page.metadata.get('sourceURL', ''))
                            
                            # Parse change tracking
                            change_tracking = None
                            if config.enable_change_tracking and hasattr(page, 'changeTracking'):
                                change_tracking = self._parse_change_tracking(page.changeTracking)
                            
                            website_data = WebsiteData(
                                chatbot_id=chatbot_id,
                                title=page.metadata.get('title', 'No Title'),
                                url=page.metadata.get('sourceURL', ''),
                                content=chunks,
                                change_tracking=change_tracking,
                                metadata={
                                    'method': 'firecrawl',
                                    'base_url': base_url,
                                    'crawled_at': datetime.now().isoformat(),
                                    'word_count': len(markdown_content.split()) if markdown_content else 0,
                                    'description': page.metadata.get('description', ''),
                                    'keywords': page.metadata.get('keywords', ''),
                                    'status_code': page.metadata.get('statusCode', 0)
                                }
                            )
                            
                            websites_data.append(website_data)
                            
                        except Exception as e:
                            logger.error(f"Error processing page: {str(e)}")
                            continue
                    
                    return websites_data
                else:
                    logger.error(f"✗ Firecrawl crawl failed: {result.get('error', 'Unknown error')}")
                    # Fallback to crawl4ai
                    method = "crawl4ai"
            
            if method == "crawl4ai":
                # Use crawl4ai for crawling (original logic)
                return await self._crawl_with_crawl4ai(base_url, chatbot_id, config, chunker)
                
        except Exception as e:
            logger.error(f"✗ Error in crawl method {method}: {str(e)}")
            # Try fallback if not already using crawl4ai
            if method != "crawl4ai":
                logger.info("Falling back to crawl4ai")
                return await self._crawl_with_crawl4ai(base_url, chatbot_id, config, chunker)
            
        return websites_data
    
    async def _crawl_with_crawl4ai(self, base_url: str, chatbot_id: str, 
                                 config: CrawlConfig, chunker: TextChunker) -> List[WebsiteData]:
        """Crawl website using crawl4ai (original implementation)"""
        # Discover URLs
        discoverer = URLDiscoverer(base_url)
        discovered_urls = set()
        
        # Get sitemap URLs
        sitemap_urls = discoverer.get_sitemap_urls()
        if sitemap_urls:
            discovered_urls.update(sitemap_urls)
            logger.info(f"📋 Found {len(sitemap_urls)} URLs from sitemap")
        
        # Always include homepage
        discovered_urls.add(base_url)
        discovered_urls.add(f"{base_url}/")
        
        # Discover more URLs by following links
        await self._discover_urls_by_crawling(discoverer, discovered_urls, config.max_depth, config.max_pages)
        
        # Apply include/exclude path filters
        if config.include_paths or config.exclude_paths:
            discovered_urls = self._filter_urls_by_paths(discovered_urls, config.include_paths, config.exclude_paths)
        
        # Limit to max_pages
        urls_to_crawl = list(discovered_urls)[:config.max_pages]
        logger.info(f"🎯 Will crawl {len(urls_to_crawl)} URLs")
        
        # Crawl all discovered URLs
        websites_data = []
        successful_count = 0
        
        async with AsyncWebCrawler() as crawler:
            for i, url in enumerate(urls_to_crawl, 1):
                try:
                    logger.info(f"[{i}/{len(urls_to_crawl)}] Crawling: {url}")
                    
                    crawler_config = CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=5,
                        page_timeout=30000
                    )
                    
                    result = await crawler.arun(url=url, config=crawler_config)
                    
                    if result.success and result.markdown:
                        chunks = chunker.chunk_text(result.markdown, url)
                        
                        website_data = WebsiteData(
                            chatbot_id=chatbot_id,
                            title=result.metadata.get('title', 'No Title'),
                            url=url,
                            content=chunks,
                            metadata={
                                'method': 'crawl4ai',
                                'base_url': base_url,
                                'crawled_at': datetime.now().isoformat(),
                                'word_count': len(result.markdown.split()) if result.markdown else 0,
                                'description': result.metadata.get('description', ''),
                                'keywords': result.metadata.get('keywords', '')
                            }
                        )
                        
                        websites_data.append(website_data)
                        successful_count += 1
                        logger.info(f"✓ Success: {len(chunks)} chunks")
                    else:
                        logger.warning(f"✗ Failed to crawl")
                
                except Exception as e:
                    logger.error(f"✗ Error: {str(e)}")
                
                # Respectful delay
                await asyncio.sleep(1)
        
        logger.info(f"🎉 Crawl complete: {successful_count}/{len(urls_to_crawl)} successful")
        return websites_data
    
    def _filter_urls_by_paths(self, urls: set, include_paths: Optional[List[str]], 
                            exclude_paths: Optional[List[str]]) -> set:
        """Filter URLs based on include/exclude path patterns"""
        filtered_urls = set()
        
        for url in urls:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Check exclude paths first
            if exclude_paths:
                excluded = False
                for exclude_pattern in exclude_paths:
                    if exclude_pattern.lower() in path:
                        excluded = True
                        break
                if excluded:
                    continue
            
            # Check include paths
            if include_paths:
                included = False
                for include_pattern in include_paths:
                    if include_pattern.lower() in path:
                        included = True
                        break
                if not included:
                    continue
            
            filtered_urls.add(url)
        
        return filtered_urls
    
    async def _discover_urls_by_crawling(self, discoverer: URLDiscoverer, 
                                       discovered_urls: set, max_depth: int, max_pages: int):
        """Discover URLs by following links"""
        crawled_urls = set()
        urls_to_explore = list(discovered_urls)[:10]  # Start with first 10
        depth = 0
        
        async with AsyncWebCrawler() as crawler:
            while (urls_to_explore and depth < max_depth and 
                   len(discovered_urls) < max_pages):
                
                logger.info(f"🌐 Discovery depth {depth + 1}: exploring {len(urls_to_explore)} URLs")
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
    
    def get_change_summary(self, websites_data: List[WebsiteData]) -> Dict[str, Any]:
        """Generate a summary of changes across all scraped pages"""
        summary = {
            'total_pages': len(websites_data),
            'new_pages': 0,
            'changed_pages': 0,
            'same_pages': 0,
            'removed_pages': 0,
            'visible_pages': 0,
            'hidden_pages': 0,
            'pages_with_changes': []
        }
        
        for data in websites_data:
            if data.change_tracking:
                ct = data.change_tracking
                
                # Count change statuses
                if ct.change_status == 'new':
                    summary['new_pages'] += 1
                elif ct.change_status == 'changed':
                    summary['changed_pages'] += 1
                    summary['pages_with_changes'].append({
                        'url': data.url,
                        'title': data.title,
                        'previous_scrape_at': ct.previous_scrape_at,
                        'has_diff': ct.diff_text is not None,
                        'has_json_changes': ct.json_changes is not None
                    })
                elif ct.change_status == 'same':
                    summary['same_pages'] += 1
                elif ct.change_status == 'removed':
                    summary['removed_pages'] += 1
                
                # Count visibility
                if ct.visibility == 'visible':
                    summary['visible_pages'] += 1
                elif ct.visibility == 'hidden':
                    summary['hidden_pages'] += 1
        
        return summary


# Enhanced convenience functions with change tracking support
async def crawl_website(base_url: str, chatbot_id: str, 
                       firecrawl_api_key: Optional[str] = None,
                       config: Optional[CrawlConfig] = None) -> List[WebsiteData]:
    """
    Convenience function to crawl a website with optional Firecrawl integration
    
    Args:
        base_url: The base URL to crawl
        chatbot_id: ID for the chatbot
        firecrawl_api_key: Optional Firecrawl API key for enhanced features
        config: Optional crawl configuration
    
    Returns:
        List of WebsiteData with change tracking information
    """
    if config is None:
        config = CrawlConfig()
    
    crawler = WebCrawler(firecrawl_api_key)
    return await crawler.crawl_website(base_url, chatbot_id, config)


async def scrape_page(url: str, chatbot_id: str,
                     firecrawl_api_key: Optional[str] = None,
                     config: Optional[ScrapeConfig] = None) -> Optional[WebsiteData]:
    """
    Convenience function to scrape a single page with optional Firecrawl integration
    
    Args:
        url: The URL to scrape
        chatbot_id: ID for the chatbot
        firecrawl_api_key: Optional Firecrawl API key for enhanced features
        config: Optional scrape configuration
    
    Returns:
        WebsiteData with change tracking information
    """
    if config is None:
        config = ScrapeConfig()
    
    crawler = WebCrawler(firecrawl_api_key)
    return await crawler.scrape_single_page(url, chatbot_id, config)


# High-level API functions
async def crawl_with_firecrawl(base_url: str, chatbot_id: str, api_key: str,
                              max_pages: int = 50, enable_change_tracking: bool = True,
                              include_paths: Optional[List[str]] = None,
                              exclude_paths: Optional[List[str]] = None,
                              max_age: Optional[int] = 3600000) -> Dict[str, Any]:
    """
    High-level API for crawling with Firecrawl
    
    Args:
        base_url: URL to crawl
        chatbot_id: Chatbot ID
        api_key: Firecrawl API key
        max_pages: Maximum pages to crawl
        enable_change_tracking: Enable change tracking
        include_paths: Paths to include
        exclude_paths: Paths to exclude
        max_age: Cache age in milliseconds
    
    Returns:
        Dictionary with crawl results and change summary
    """
    config = CrawlConfig(
        method="firecrawl",
        max_pages=max_pages,
        enable_change_tracking=enable_change_tracking,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_age=max_age
    )
    
    crawler = WebCrawler(api_key)
    websites_data = await crawler.crawl_website(base_url, chatbot_id, config)
    
    # Generate change summary
    change_summary = crawler.get_change_summary(websites_data)
    
    return {
        'success': True,
        'base_url': base_url,
        'total_pages': len(websites_data),
        'change_summary': change_summary,
        'data': [data.to_dict() for data in websites_data],
        'crawled_at': datetime.now().isoformat()
    }


async def scrape_with_firecrawl(url: str, chatbot_id: str, api_key: str,
                               enable_change_tracking: bool = True,
                               only_main_content: bool = False,
                               max_age: Optional[int] = 3600000) -> Dict[str, Any]:
    """
    High-level API for scraping with Firecrawl
    
    Args:
        url: URL to scrape
        chatbot_id: Chatbot ID
        api_key: Firecrawl API key
        enable_change_tracking: Enable change tracking
        only_main_content: Extract only main content
        max_age: Cache age in milliseconds
    
    Returns:
        Dictionary with scrape results and change information
    """
    config = ScrapeConfig(
        method="firecrawl",
        enable_change_tracking=enable_change_tracking,
        only_main_content=only_main_content,
        max_age=max_age
    )
    
    crawler = WebCrawler(api_key)
    website_data = await crawler.scrape_single_page(url, chatbot_id, config)
    
    if website_data:
        return {
            'success': True,
            'data': website_data.to_dict(),
            'change_tracking': website_data.change_tracking.to_dict() if website_data.change_tracking else None,
            'scraped_at': datetime.now().isoformat()
        }
    else:
        return {
            'success': False,
            'error': 'Failed to scrape page',
            'url': url
        }


async def crawl_with_crawl4ai(base_url: str, chatbot_id: str,
                             max_pages: int = 50, max_depth: int = 2,
                             chunk_size: int = 1000) -> Dict[str, Any]:
    """
    High-level API for crawling with crawl4ai
    
    Args:
        base_url: URL to crawl
        chatbot_id: Chatbot ID
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        chunk_size: Text chunk size
    
    Returns:
        Dictionary with crawl results
    """
    config = CrawlConfig(
        method="crawl4ai",
        max_pages=max_pages,
        max_depth=max_depth,
        chunk_size=chunk_size,
        enable_change_tracking=False  # Not supported with crawl4ai
    )
    
    crawler = WebCrawler()
    websites_data = await crawler.crawl_website(base_url, chatbot_id, config)
    
    return {
        'success': True,
        'base_url': base_url,
        'total_pages': len(websites_data),
        'data': [data.to_dict() for data in websites_data],
        'crawled_at': datetime.now().isoformat()
    }


async def scrape_with_crawl4ai(url: str, chatbot_id: str,
                              chunk_size: int = 1000) -> Dict[str, Any]:
    """
    High-level API for scraping with crawl4ai
    
    Args:
        url: URL to scrape
        chatbot_id: Chatbot ID
        chunk_size: Text chunk size
    
    Returns:
        Dictionary with scrape results
    """
    config = ScrapeConfig(
        method="crawl4ai",
        chunk_size=chunk_size,
        enable_change_tracking=False  # Not supported with crawl4ai
    )
    
    crawler = WebCrawler()
    website_data = await crawler.scrape_single_page(url, chatbot_id, config)
    
    if website_data:
        return {
            'success': True,
            'data': website_data.to_dict(),
            'scraped_at': datetime.now().isoformat()
        }
    else:
        return {
            'success': False,
            'error': 'Failed to scrape page',
            'url': url
        }


# Batch processing functions
async def batch_scrape_with_firecrawl(urls: List[str], chatbot_id: str, api_key: str,
                                     enable_change_tracking: bool = True,
                                     max_age: Optional[int] = 3600000) -> Dict[str, Any]:
    """
    Batch scrape multiple URLs with Firecrawl
    
    Args:
        urls: List of URLs to scrape
        chatbot_id: Chatbot ID
        api_key: Firecrawl API key
        enable_change_tracking: Enable change tracking
        max_age: Cache age in milliseconds
    
    Returns:
        Dictionary with batch scrape results
    """
    if not FIRECRAWL_AVAILABLE:
        return {
            'success': False,
            'error': 'Firecrawl not available'
        }
    
    try:
        app = FirecrawlApp(api_key=api_key)
        
        scrape_params = {
            'formats': ['markdown', 'changeTracking'] if enable_change_tracking else ['markdown'],
            'maxAge': max_age
        }
        
        if enable_change_tracking:
            scrape_params['changeTrackingOptions'] = {
                'modes': ['git-diff', 'json'],
                'tag': 'batch'
            }
        
        result = app.batch_scrape_urls(urls, **scrape_params)
        
        if result.success:
            chunker = TextChunker()
            websites_data = []
            
            for page in result.data:
                try:
                    markdown_content = getattr(page, 'markdown', '')
                    chunks = chunker.chunk_text(markdown_content, page.metadata.get('sourceURL', ''))
                    
                    # Parse change tracking
                    change_tracking = None
                    if enable_change_tracking and hasattr(page, 'changeTracking'):
                        crawler = WebCrawler()
                        change_tracking = crawler._parse_change_tracking(page.changeTracking)
                    
                    website_data = WebsiteData(
                        chatbot_id=chatbot_id,
                        title=page.metadata.get('title', 'No Title'),
                        url=page.metadata.get('sourceURL', ''),
                        content=chunks,
                        change_tracking=change_tracking,
                        metadata={
                            'method': 'firecrawl_batch',
                            'scraped_at': datetime.now().isoformat(),
                            'word_count': len(markdown_content.split()) if markdown_content else 0,
                            'description': page.metadata.get('description', ''),
                            'status_code': page.metadata.get('statusCode', 0)
                        }
                    )
                    
                    websites_data.append(website_data)
                    
                except Exception as e:
                    logger.error(f"Error processing batch page: {str(e)}")
                    continue
            
            # Generate change summary
            crawler = WebCrawler()
            change_summary = crawler.get_change_summary(websites_data)
            
            return {
                'success': True,
                'total_urls': len(urls),
                'successful_scrapes': len(websites_data),
                'credits_used': getattr(result, 'creditsUsed', 0),
                'change_summary': change_summary,
                'data': [data.to_dict() for data in websites_data],
                'scraped_at': datetime.now().isoformat()
            }
        else:
            return {
                'success': False,
                'error': 'Batch scrape failed'
            }
            
    except Exception as e:
        logger.error(f"Batch scrape error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Get API key from environment or use placeholder
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not firecrawl_api_key:
            print("⚠️  No FIRECRAWL_API_KEY found in environment variables")
            print("Set it with: export FIRECRAWL_API_KEY='your-api-key'")
            print("Falling back to crawl4ai for demonstration...")
        
        chatbot_id = "demo_chatbot"
        
        # Example 1: Crawl website with Firecrawl (if available)
        print("\n=== Example 1: Website Crawling ===")
        if firecrawl_api_key:
            result = await crawl_with_firecrawl(
                base_url="https://docs.firecrawl.dev",
                chatbot_id=chatbot_id,
                api_key=firecrawl_api_key,
                max_pages=5,
                enable_change_tracking=True
            )
        else:
            result = await crawl_with_crawl4ai(
                base_url="https://docs.firecrawl.dev",
                chatbot_id=chatbot_id,
                max_pages=5
            )
        
        if result['success']:
            print(f"✓ Crawled {result['total_pages']} pages")
            if 'change_summary' in result:
                print(f"📊 Change Summary: {result['change_summary']}")
            
            # Show pages with changes
            for page_data in result['data']:
                if page_data.get('change_tracking'):
                    ct = page_data['change_tracking']
                    if ct['change_status'] in ['new', 'changed']:
                        print(f"  🔄 {page_data['url']}: {ct['change_status']}")
                else:
                    # For crawl4ai results without change tracking
                    print(f"  📄 {page_data['url']}: crawled successfully")
        else:
            print(f"✗ Crawl failed: {result.get('error', 'Unknown error')}")
        
        # Example 2: Single page scraping
        print("\n=== Example 2: Single Page Scraping ===")
        if firecrawl_api_key:
            result = await scrape_with_firecrawl(
                url="https://firecrawl.dev",
                chatbot_id=chatbot_id,
                api_key=firecrawl_api_key,
                enable_change_tracking=True
            )
        else:
            result = await scrape_with_crawl4ai(
                url="https://firecrawl.dev",
                chatbot_id=chatbot_id
            )
        
        if result['success']:
            page_data = result['data']
            print(f"✓ Scraped: {page_data['title']}")
            print(f"📄 Chunks: {len(page_data['content'])}")
            print(f"📝 Word Count: {page_data['metadata'].get('word_count', 0)}")
            
            if result.get('change_tracking'):
                ct = result['change_tracking']
                print(f"🔄 Change Status: {ct['change_status']}")
                if ct['change_status'] == 'changed':
                    print("  📝 Changes detected!")
                    if ct.get('previous_scrape_at'):
                        print(f"  🕒 Previous scrape: {ct['previous_scrape_at']}")
            else:
                print("📊 Method: crawl4ai (no change tracking)")
        else:
            print(f"✗ Scrape failed: {result.get('error', 'Unknown error')}")
        
        # Example 3: Batch scraping (Firecrawl only)
        if firecrawl_api_key:
            print("\n=== Example 3: Batch Scraping ===")
            urls = [
                "https://firecrawl.dev",
                "https://docs.firecrawl.dev"
            ]
            
            result = await batch_scrape_with_firecrawl(
                urls=urls,
                chatbot_id=chatbot_id,
                api_key=firecrawl_api_key,
                enable_change_tracking=True
            )
            
            if result['success']:
                print(f"✓ Batch scraped {result['successful_scrapes']}/{result['total_urls']} URLs")
                print(f"💳 Credits used: {result.get('credits_used', 0)}")
                if 'change_summary' in result:
                    cs = result['change_summary']
                    print(f"📊 Change Summary:")
                    print(f"   - New pages: {cs.get('new_pages', 0)}")
                    print(f"   - Changed pages: {cs.get('changed_pages', 0)}")
                    print(f"   - Same pages: {cs.get('same_pages', 0)}")
            else:
                print(f"✗ Batch scrape failed: {result.get('error', 'Unknown error')}")
        
        # Example 4: Advanced configuration
        print("\n=== Example 4: Advanced Configuration ===")
        try:
            # Create custom configuration
            custom_config = CrawlConfig(
                method="auto",
                max_pages=3,
                max_depth=1,
                chunk_size=800,
                chunk_overlap=150,
                enable_change_tracking=bool(firecrawl_api_key),
                exclude_paths=["/api/*", "/admin/*"]
            )
            
            # Use the WebCrawler class directly for advanced control
            crawler = WebCrawler(firecrawl_api_key)
            websites_data = await crawler.crawl_website(
                base_url="https://firecrawl.dev",
                chatbot_id=chatbot_id,
                config=custom_config
            )
            
            print(f"✓ Advanced crawl completed: {len(websites_data)} pages")
            
            # Show detailed information
            for website_data in websites_data:
                print(f"  📄 {website_data.title}")
                print(f"     URL: {website_data.url}")
                print(f"     Chunks: {len(website_data.content)}")
                print(f"     Method: {website_data.metadata.get('method', 'unknown')}")
                
                if website_data.change_tracking:
                    print(f"     Change Status: {website_data.change_tracking.change_status}")
                
                # Show first chunk preview
                if website_data.content:
                    first_chunk = website_data.content[0]
                    preview = first_chunk.content[:100] + "..." if len(first_chunk.content) > 100 else first_chunk.content
                    print(f"     Preview: {preview}")
                print()
                
        except Exception as e:
            print(f"✗ Advanced crawl failed: {str(e)}")
        
        # Example 5: Configuration testing
        print("\n=== Example 5: Configuration Testing ===")
        
        # Test different chunk sizes
        chunk_sizes = [500, 1000, 1500]
        test_text = "This is a test paragraph. " * 50  # Create a longer text
        
        for chunk_size in chunk_sizes:
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=100)
            chunks = chunker.chunk_text(test_text, "https://test.com")
            print(f"📏 Chunk size {chunk_size}: {len(chunks)} chunks")
            
            if chunks:
                avg_tokens = sum(chunk.metadata['chunk_token_count'] for chunk in chunks) / len(chunks)
                print(f"   Average tokens per chunk: {avg_tokens:.1f}")
        
        print("\n=== Demo Complete ===")
        print("\n💡 Tips:")
        print("- Set FIRECRAWL_API_KEY environment variable for enhanced features")
        print("- Use change tracking to only process new/changed content")
        print("- Configure chunk sizes based on your LLM context window")
        print("- Use include/exclude paths for targeted crawling")
        
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


# Additional utility functions for advanced users
def create_custom_crawler(api_key: Optional[str] = None, 
                         default_chunk_size: int = 1000,
                         default_overlap: int = 200) -> WebCrawler:
    """
    Create a customized WebCrawler instance
    
    Args:
        api_key: Optional Firecrawl API key
        default_chunk_size: Default chunk size for text processing
        default_overlap: Default chunk overlap
    
    Returns:
        Configured WebCrawler instance
    """
    crawler = WebCrawler(api_key)
    # You can add custom initialization here
    return crawler


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def estimate_crawl_cost(base_url: str, max_pages: int = 50, 
                       enable_firecrawl: bool = True) -> Dict[str, Any]:
    """
    Estimate the cost and time for a crawl operation
    
    Args:
        base_url: Base URL to crawl
        max_pages: Maximum pages to crawl
        enable_firecrawl: Whether Firecrawl will be used
        
    Returns:
        Dictionary with cost and time estimates
    """
    # Basic estimates (these would need to be calibrated based on actual usage)
    firecrawl_credits_per_page = 1
    crawl4ai_time_per_page = 2  # seconds
    firecrawl_time_per_page = 0.5  # seconds (due to caching and optimization)
    
    if enable_firecrawl:
        estimated_credits = max_pages * firecrawl_credits_per_page
        estimated_time = max_pages * firecrawl_time_per_page
        method = "firecrawl"
    else:
        estimated_credits = 0  # crawl4ai is free
        estimated_time = max_pages * crawl4ai_time_per_page
        method = "crawl4ai"
    
    return {
        'base_url': base_url,
        'max_pages': max_pages,
        'method': method,
        'estimated_credits': estimated_credits,
        'estimated_time_seconds': estimated_time,
        'estimated_time_minutes': estimated_time / 60,
        'features': {
            'change_tracking': enable_firecrawl,
            'caching': enable_firecrawl,
            'advanced_extraction': enable_firecrawl
        }
    }


# Export all public functions and classes
__all__ = [
    # Data classes
    'ChunkData',
    'ChangeTrackingData', 
    'WebsiteData',
    'CrawlConfig',
    'ScrapeConfig',
    
    # Main classes
    'TextChunker',
    'URLDiscoverer',
    'FirecrawlIntegration',
    'WebCrawler',
    
    # Convenience functions
    'crawl_website',
    'scrape_page',
    
    # High-level API functions
    'crawl_with_firecrawl',
    'scrape_with_firecrawl',
    'crawl_with_crawl4ai', 
    'scrape_with_crawl4ai',
    'batch_scrape_with_firecrawl',
    
    # Utility functions
    'create_custom_crawler',
    'validate_url',
    'estimate_crawl_cost'
]

# Version information
__version__ = "2.0.0"
__author__ = "Enhanced Web Crawler"
__description__ = "A comprehensive web crawler with Firecrawl integration and change tracking"