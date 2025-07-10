#!/usr/bin/env python3
"""
Article content extraction and summarization using newspaper3k and Hugging Face BART
"""

import requests
import logging
from typing import Optional, Dict, Any
import time

# Try to import optional dependencies
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
    print("✅ newspaper3k available")
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("⚠️ newspaper3k not installed. Using fallback content extraction.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✅ transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed. Install with: pip install transformers")

# Fallback imports for content extraction
try:
    from bs4 import BeautifulSoup
    import re
    BEAUTIFULSOUP_AVAILABLE = True
    print("✅ BeautifulSoup available for fallback extraction")
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ BeautifulSoup not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleSummarizer:
    def __init__(self):
        """Initialize the summarizer with lazy loading for AI models"""
        self.summarizer = None
        self.sentiment_analyzer = None
        self._summarizer_loaded = False
        self._sentiment_loaded = False
        
        # Content extraction available if newspaper3k OR BeautifulSoup is available
        self.content_extraction_available = NEWSPAPER_AVAILABLE or BEAUTIFULSOUP_AVAILABLE
        self.dependencies_available = self.content_extraction_available and TRANSFORMERS_AVAILABLE
        
        if not self.content_extraction_available:
            logger.warning("No content extraction method available")
            logger.warning("Install newspaper3k: pip install newspaper3k")
            
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available for AI summarization")
            logger.warning("Install transformers: pip install transformers")
            
        if not self.dependencies_available:
            logger.warning("Required dependencies not available for summarization")
    
    def _load_summarizer(self):
        """Lazy load the BART summarization model"""
        if self._summarizer_loaded:
            return self.summarizer is not None
            
        if not TRANSFORMERS_AVAILABLE:
            self._summarizer_loaded = True
            return False
            
        try:
            logger.info("Loading summarization model...")
            # Try different models in order of preference (less hallucination-prone)
            models_to_try = [
                "facebook/bart-large-cnn",
                "sshleifer/distilbart-cnn-12-6",  # Smaller, faster, less prone to hallucination
                "t5-small"  # T5 as fallback
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Trying model: {model_name}")
                    self.summarizer = pipeline(
                        "summarization",
                        model=model_name,
                        tokenizer=model_name,
                        device=-1,  # Use CPU (-1) or GPU (0)
                        framework="pt"
                    )
                    logger.info(f"Model {model_name} loaded successfully")
                    self._summarizer_loaded = True
                    return True
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue
            
            # If all models fail
            logger.error("All summarization models failed to load")
            self.summarizer = None
            self._summarizer_loaded = True
            return False
        except Exception as e:
            logger.error(f"Error loading BART model: {e}")
            self.summarizer = None
            self._summarizer_loaded = True
            return False
    
    def _load_sentiment_analyzer(self):
        """Lazy load the sentiment analysis model"""
        if self._sentiment_loaded:
            return self.sentiment_analyzer is not None
            
        if not TRANSFORMERS_AVAILABLE:
            self._sentiment_loaded = True
            return False
            
        try:
            logger.info("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU (-1) or GPU (0)
            )
            logger.info("Sentiment analysis model loaded successfully")
            self._sentiment_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {e}")
            self.sentiment_analyzer = None
            self._sentiment_loaded = True
            return False
    
    def extract_article_content(self, url: str) -> Optional[str]:
        """
        Extract full article content from URL using newspaper3k or BeautifulSoup fallback
        
        Args:
            url: Article URL to extract content from
            
        Returns:
            Extracted article text or None if extraction fails
        """
        if not self.content_extraction_available:
            logger.error("No content extraction method available")
            return None
        
        content = None
        
        # Try newspaper3k first (preferred method)
        if NEWSPAPER_AVAILABLE:
            content = self._extract_with_newspaper(url)
            
        # Fallback to BeautifulSoup if newspaper3k fails
        if not content and BEAUTIFULSOUP_AVAILABLE:
            content = self._extract_with_beautifulsoup(url)
            
        # If we still don't have content, try to extract from the URL's metadata
        if not content:
            content = self._extract_from_metadata(url)
            
        return content
        
    def _extract_from_metadata(self, url: str) -> Optional[str]:
        """
        Extract content from article metadata when direct extraction fails
        
        Args:
            url: Article URL
            
        Returns:
            Extracted content or None if extraction fails
        """
        try:
            logger.info(f"Attempting to extract metadata from URL: {url}")
            
            # Try to find the article in our cache first
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from app import articles_cache
                
                # Find article by URL
                for article in articles_cache:
                    if article.get('url') == url:
                        logger.info(f"Found article in cache: {article.get('title')}")
                        
                        # Construct content from available metadata
                        content_parts = []
                        
                        # Add title
                        if article.get('title'):
                            content_parts.append(f"# {article['title']}")
                        
                        # Add summary/description
                        if article.get('summary'):
                            content_parts.append(article['summary'])
                        
                        # Add content if available
                        if article.get('content'):
                            content_parts.append(article['content'])
                        
                        # Add author
                        if article.get('author') and article['author'] != 'Unknown':
                            content_parts.append(f"Author: {article['author']}")
                        
                        # Add source
                        if article.get('source') and article['source'] != 'Unknown':
                            content_parts.append(f"Source: {article['source']}")
                        
                        # Combine all parts
                        combined_content = "\n\n".join(content_parts)
                        
                        if len(combined_content) > 200:  # Minimum content length
                            logger.info(f"Successfully constructed content from cache: {len(combined_content)} chars")
                            return combined_content
            except Exception as cache_error:
                logger.warning(f"Error accessing cache: {cache_error}")
            
            # If we can't get from cache, make a simple request to get metadata
            if NEWSPAPER_AVAILABLE:
                from newspaper import Article
                article = Article(url)
                # Only download and parse metadata (no full download)
                article.download()
                article.parse()
                
                # Get metadata
                title = article.title
                text = article.text
                
                # If we have some content, return it
                if title and text and len(text) > 100:
                    logger.info(f"Extracted metadata content: {len(text)} chars")
                    return f"# {title}\n\n{text}"
            
            # Last resort: try to get Open Graph metadata
            if BEAUTIFULSOUP_AVAILABLE:
                import requests
                from bs4 import BeautifulSoup
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract Open Graph metadata
                og_title = soup.find("meta", property="og:title")
                og_description = soup.find("meta", property="og:description")
                
                if og_title or og_description:
                    content_parts = []
                    
                    if og_title and og_title.get("content"):
                        content_parts.append(f"# {og_title['content']}")
                    
                    if og_description and og_description.get("content"):
                        content_parts.append(og_description["content"])
                    
                    # Try to get the first few paragraphs
                    paragraphs = soup.find_all("p")[:5]  # Get first 5 paragraphs
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 50:  # Only include substantial paragraphs
                            content_parts.append(text)
                    
                    combined_content = "\n\n".join(content_parts)
                    
                    if len(combined_content) > 200:  # Minimum content length
                        logger.info(f"Extracted Open Graph metadata: {len(combined_content)} chars")
                        return combined_content
            
            logger.warning(f"Failed to extract any content from {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {e}")
            return None
    
    def _extract_with_newspaper(self, url: str) -> Optional[str]:
        """Extract content using newspaper3k"""
        try:
            logger.info(f"Extracting content with newspaper3k from: {url}")
            
            # Create Article object and download
            article = Article(url)
            article.download()
            
            # Parse the article
            article.parse()
            
            # Get the main content
            content = article.text
            
            if content and len(content.strip()) > 100:  # Minimum content length
                logger.info(f"Successfully extracted {len(content)} characters from {url}")
                logger.info(f"Article title: {article.title}")
                logger.info(f"Article authors: {article.authors}")
                logger.info(f"Publish date: {article.publish_date}")
                return content.strip()
            else:
                logger.warning(f"Insufficient content extracted from {url}")
                logger.warning(f"Content length: {len(content) if content else 0}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content with newspaper3k from {url}: {e}")
            return None
    
    def _extract_with_beautifulsoup(self, url: str) -> Optional[str]:
        """Fallback content extraction using BeautifulSoup"""
        try:
            logger.info(f"Extracting content with BeautifulSoup fallback from: {url}")
            
            # Download the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article',
                '[role="main"]',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.story-body',
                '.article-body'
            ]
            
            content_text = ""
            
            # Try each selector
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text(separator=' ', strip=True)
                        if len(text) > len(content_text):
                            content_text = text
                    break
            
            # If no specific content area found, get all paragraphs
            if not content_text or len(content_text) < 200:
                paragraphs = soup.find_all('p')
                content_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Clean up the text
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            if content_text and len(content_text) > 100:
                logger.info(f"Successfully extracted {len(content_text)} characters with BeautifulSoup")
                return content_text
            else:
                logger.warning(f"Insufficient content extracted with BeautifulSoup: {len(content_text)} chars")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content with BeautifulSoup from {url}: {e}")
            return None
    
    def _preprocess_text_for_summarization(self, text: str) -> str:
        """
        Preprocess text to improve summarization quality and reduce hallucination
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs and email addresses that might confuse the model
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove common navigation/UI text that might appear in scraped content
        ui_patterns = [
            r'click here',
            r'read more',
            r'subscribe',
            r'newsletter',
            r'advertisement',
            r'sponsored content',
            r'share this article',
            r'follow us',
            r'related articles',
            r'trending now'
        ]
        
        for pattern in ui_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up any double spaces created by removals
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extractive_summarization_fallback(self, text: str, max_sentences: int = 3) -> str:
        """
        Simple extractive summarization as fallback when AI model fails or hallucinates
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Score sentences based on position and length (simple heuristic)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Earlier sentences get higher scores
            position_score = 1.0 - (i / len(sentences)) * 0.5
            # Moderate length sentences get higher scores
            length_score = min(len(sentence) / 100, 1.0)
            total_score = position_score + length_score
            scored_sentences.append((sentence, total_score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        # Reorder sentences to maintain original order
        original_order_summary = []
        for sentence in sentences:
            if sentence in top_sentences:
                original_order_summary.append(sentence)
                if len(original_order_summary) >= max_sentences:
                    break
        
        return '. '.join(original_order_summary) + '.'
    
    def _validate_summary_against_source(self, summary: str, source_text: str) -> bool:
        """
        Enhanced validation to check if summary contains information not in source
        """
        import re
        
        # Extract key entities (numbers, proper nouns) from both texts
        def extract_entities(text):
            # Extract numbers (especially those that might be statistics)
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
            # Extract capitalized words (potential proper nouns, but be more selective)
            proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text)
            # Extract specific patterns that might indicate facts
            dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text)
            return {
                'numbers': set(numbers),
                'proper_nouns': set(proper_nouns),
                'dates': set(dates)
            }
        
        summary_entities = extract_entities(summary)
        source_entities = extract_entities(source_text)
        
        # Check for entities in summary that aren't in source
        hallucinated_numbers = summary_entities['numbers'] - source_entities['numbers']
        hallucinated_nouns = summary_entities['proper_nouns'] - source_entities['proper_nouns']
        hallucinated_dates = summary_entities['dates'] - source_entities['dates']
        
        # Filter out common words that might be capitalized
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'So', 
            'For', 'At', 'In', 'On', 'To', 'From', 'With', 'By', 'As', 'Of', 'Is', 'Are',
            'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had', 'Do', 'Does', 'Did',
            'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Must', 'Can', 'Cannot',
            'Article', 'News', 'Report', 'Story', 'Today', 'Yesterday', 'Tomorrow', 'Now',
            'Here', 'There', 'Where', 'When', 'How', 'Why', 'What', 'Who', 'Which'
        }
        hallucinated_nouns = hallucinated_nouns - common_words
        
        # Count total hallucinated entities
        total_hallucinated = len(hallucinated_numbers) + len(hallucinated_nouns) + len(hallucinated_dates)
        
        # Be more strict about numbers and dates (these are often factual)
        critical_hallucinations = len(hallucinated_numbers) + len(hallucinated_dates)
        
        if critical_hallucinations > 0:
            logger.warning(f"Critical hallucination detected:")
            if hallucinated_numbers:
                logger.warning(f"  Numbers not in source: {hallucinated_numbers}")
            if hallucinated_dates:
                logger.warning(f"  Dates not in source: {hallucinated_dates}")
            return False
        
        # Be more strict with proper nouns that could be names or places
        if len(hallucinated_nouns) > 1:
            logger.warning(f"Proper nouns not in source: {hallucinated_nouns}")
            return False
        
        # Additional check: look for specific phrases that often indicate hallucination
        hallucination_phrases = [
            r'\b\d+\s+(?:people|children|adults|victims|casualties)\b',
            r'\b(?:confirmed|reported)\s+dead\b',
            r'\b(?:flash|major|severe)\s+floods?\b',
            r'\bat least\s+\d+\b',
            r'\bmore than\s+\d+\s+(?:people|children|adults)\b'
        ]
        
        for phrase_pattern in hallucination_phrases:
            if re.search(phrase_pattern, summary, re.IGNORECASE):
                if not re.search(phrase_pattern, source_text, re.IGNORECASE):
                    logger.warning(f"Suspicious phrase found in summary but not source: {phrase_pattern}")
                    return False
        
        return True

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> Optional[str]:
        """
        Summarize text using BART model with hallucination prevention
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summary text or None if summarization fails
        """
        try:
            # Preprocess text to reduce hallucination risk
            processed_text = self._preprocess_text_for_summarization(text)
            
            # If text is very short, return it as-is
            if len(processed_text.split()) < 50:
                return processed_text
            
            # Lazy load the summarizer
            if not self._load_summarizer():
                logger.warning("BART summarizer not available, using extractive fallback")
                return self._extractive_summarization_fallback(processed_text)
            
            # Truncate text if too long (BART has token limits)
            max_input_length = 800  # Reduced from 1024 to prevent hallucination
            if len(processed_text.split()) > max_input_length:
                processed_text = ' '.join(processed_text.split()[:max_input_length])
                logger.info(f"Truncated input text to {max_input_length} words")
            
            logger.info(f"Summarizing text of {len(processed_text)} characters...")
            
            # Generate summary with more conservative parameters
            summary_result = self.summarizer(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
                no_repeat_ngram_size=3,  # Prevent repetition
                early_stopping=True      # Stop when good summary is found
            )
            
            if summary_result and len(summary_result) > 0:
                summary = summary_result[0]['summary_text']
                
                # Validate summary against source to detect hallucination
                if not self._validate_summary_against_source(summary, processed_text):
                    logger.warning("Potential hallucination detected, using extractive fallback")
                    return self._extractive_summarization_fallback(processed_text)
                
                logger.info(f"Generated summary of {len(summary)} characters")
                return summary
            else:
                logger.warning("No summary generated, using extractive fallback")
                return self._extractive_summarization_fallback(processed_text)
                
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            logger.info("Falling back to extractive summarization")
            return self._extractive_summarization_fallback(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using RoBERTa model with fallback to rule-based analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Try to use the transformer model first
            if self._load_sentiment_analyzer():
                # Truncate text if too long
                max_input_length = 512  # DistilBERT's max input length
                if len(text.split()) > max_input_length:
                    text = ' '.join(text.split()[:max_input_length])
                    logger.info(f"Truncated input text to {max_input_length} words for sentiment analysis")
                
                logger.info(f"Analyzing sentiment of text with {len(text)} characters using transformer model...")
                
                # Analyze sentiment
                result = self.sentiment_analyzer(text)
                
                if result and len(result) > 0:
                    sentiment_result = result[0]
                    
                    # Map the model's labels to our expected format
                    label_mapping = {
                        'NEGATIVE': 'negative',
                        'POSITIVE': 'positive',
                        # Fallback mappings for other potential label formats
                        'LABEL_0': 'negative',  # Some models use LABEL_0 for negative
                        'LABEL_1': 'positive',  # Some models use LABEL_1 for positive
                        'NEUTRAL': 'neutral'
                    }
                    
                    original_label = sentiment_result['label']
                    mapped_label = label_mapping.get(original_label, 'neutral')
                    confidence = sentiment_result['score']
                    
                    # Convert confidence to sentiment score (-1 to 1)
                    if mapped_label == 'positive':
                        score = confidence
                    elif mapped_label == 'negative':
                        score = -confidence
                    else:  # neutral
                        score = 0.0
                    
                    logger.info(f"Transformer sentiment analysis result: {mapped_label} (confidence: {confidence:.3f})")
                    
                    return {
                        "label": mapped_label,
                        "score": round(score, 3),
                        "confidence": round(confidence, 3)
                    }
            
            # Fallback to rule-based sentiment analysis
            logger.info("Using fallback rule-based sentiment analysis...")
            return self._rule_based_sentiment_analysis(text)
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment with transformer model: {e}")
            logger.info("Falling back to rule-based sentiment analysis...")
            return self._rule_based_sentiment_analysis(text)
    
    def _rule_based_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Fallback rule-based sentiment analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Import the enhanced sentiment analysis from app.py
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from app import analyze_sentiment
            
            result = analyze_sentiment(text)
            logger.info(f"Rule-based sentiment analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in rule-based sentiment analysis: {e}")
            return {
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.1
            }
    
    def summarize_article_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract article content from URL and generate summary
        
        Args:
            url: Article URL
            
        Returns:
            Dictionary with summary result and metadata
        """
        start_time = time.time()
        
        # Check if dependencies are available
        if not self.dependencies_available:
            missing = []
            if not self.content_extraction_available:
                missing.append("newspaper3k (or BeautifulSoup)")
            if not TRANSFORMERS_AVAILABLE:
                missing.append("transformers")
            
            install_cmd = "pip install transformers"
            if not self.content_extraction_available:
                install_cmd = "pip install newspaper3k transformers"
            
            return {
                "success": False,
                "summary": "Unable to summarize this article.",
                "error": f"Required dependencies not installed: {', '.join(missing)}. Install with: {install_cmd}",
                "url": url,
                "processing_time": round(time.time() - start_time, 2)
            }
        
        try:
            # Step 1: Extract article content
            content = self.extract_article_content(url)
            
            if not content:
                # Try to get article from cache to provide a basic summary
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    from app import articles_cache
                    
                    # Find article by URL
                    for article in articles_cache:
                        if article.get('url') == url:
                            logger.info(f"Found article in cache: {article.get('title')}")
                            
                            # Use the article's existing summary if available
                            if article.get('summary') and len(article.get('summary', '')) > 50:
                                return {
                                    "success": True,
                                    "summary": f"From article metadata: {article['summary']}",
                                    "sentiment": article.get('sentiment', {"label": "neutral", "score": 0, "confidence": 0.1}),
                                    "url": url,
                                    "content_length": len(article.get('content', '')),
                                    "summary_length": len(article.get('summary', '')),
                                    "processing_time": round(time.time() - start_time, 2),
                                    "note": "This summary was generated from article metadata as full content extraction failed."
                                }
                except Exception as cache_error:
                    logger.warning(f"Error accessing cache: {cache_error}")
                
                # If we can't get from cache either, return error
                # Try to determine if it's a paywall or just insufficient content
                is_paywall = False
                try:
                    # Check for common paywall indicators
                    import requests
                    from bs4 import BeautifulSoup
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Common paywall indicators in text
                    paywall_phrases = [
                        'subscribe to continue', 'subscription required', 'premium content',
                        'subscribe now', 'premium article', 'for subscribers only',
                        'sign up to read', 'premium access', 'paid subscribers',
                        'create an account', 'register to continue', 'login to continue',
                        'members only', 'premium membership', 'paid content'
                    ]
                    
                    page_text = soup.get_text().lower()
                    for phrase in paywall_phrases:
                        if phrase in page_text:
                            is_paywall = True
                            break
                            
                    # Check for login/subscription forms
                    paywall_elements = soup.select('form.paywall, div.paywall, .subscription-required, .premium-content, .login-required')
                    if paywall_elements:
                        is_paywall = True
                        
                except Exception as e:
                    logger.warning(f"Error checking for paywall: {e}")
                
                error_message = "Could not extract article content. "
                if is_paywall:
                    error_message += "This article is behind a paywall and requires a subscription to access."
                else:
                    error_message += "The article may have insufficient content or uses anti-scraping measures."
                
                return {
                    "success": False,
                    "summary": "Unable to summarize this article.",
                    "error": error_message,
                    "url": url,
                    "is_paywall": is_paywall,
                    "processing_time": round(time.time() - start_time, 2)
                }
            
            # Step 2: Generate summary
            summary = self.summarize_text(content)
            
            if not summary:
                # If AI summarization fails, use a simple extractive approach
                try:
                    # Extract first 2-3 sentences as a basic summary
                    import re
                    sentences = re.split(r'[.!?]+', content)
                    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                    
                    if sentences:
                        basic_summary = '. '.join(sentences[:3]) + '.'
                        logger.info(f"Using basic extractive summary: {len(basic_summary)} chars")
                        
                        # Step 3: Analyze sentiment of the content
                        sentiment = self.analyze_sentiment(content)
                        
                        return {
                            "success": True,
                            "summary": basic_summary,
                            "sentiment": sentiment,
                            "url": url,
                            "content_length": len(content),
                            "summary_length": len(basic_summary),
                            "processing_time": round(time.time() - start_time, 2),
                            "note": "This is a basic extractive summary as AI summarization failed."
                        }
                except Exception as extract_error:
                    logger.warning(f"Error creating basic summary: {extract_error}")
                
                return {
                    "success": False,
                    "summary": "Unable to summarize this article.",
                    "error": "Content extraction successful but summarization failed.",
                    "url": url,
                    "content_length": len(content),
                    "processing_time": round(time.time() - start_time, 2)
                }
            
            # Step 3: Analyze sentiment of the content
            sentiment = self.analyze_sentiment(content)
            
            # Step 4: Return successful result
            return {
                "success": True,
                "summary": summary,
                "sentiment": sentiment,
                "url": url,
                "content_length": len(content),
                "summary_length": len(summary),
                "processing_time": round(time.time() - start_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error processing article {url}: {e}")
            return {
                "success": False,
                "summary": "Unable to summarize this article.",
                "error": f"Processing error: {str(e)}",
                "url": url,
                "processing_time": round(time.time() - start_time, 2)
            }

# Global summarizer instance
_summarizer_instance = None

def get_summarizer():
    """Get or create the global summarizer instance"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ArticleSummarizer()
    return _summarizer_instance

def summarize_article_url(url: str) -> Dict[str, Any]:
    """
    Convenience function to summarize an article from URL
    
    Args:
        url: Article URL
        
    Returns:
        Summary result dictionary
    """
    summarizer = get_summarizer()
    return summarizer.summarize_article_from_url(url)

# Test function
if __name__ == "__main__":
    # Test with a sample URL
    test_url = "https://www.bbc.com/news"
    
    print("🤖 Testing Article Summarization...")
    print(f"URL: {test_url}")
    
    result = summarize_article_url(test_url)
    
    print(f"\nResult:")
    print(f"Success: {result['success']}")
    print(f"Summary: {result['summary']}")
    print(f"Processing Time: {result['processing_time']}s")
    
    if result['success']:
        print(f"Content Length: {result['content_length']} chars")
        print(f"Summary Length: {result['summary_length']} chars")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")