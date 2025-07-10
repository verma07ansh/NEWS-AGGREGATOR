from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
from datetime import datetime, timedelta
import uuid
import hashlib
from dotenv import load_dotenv
import logging

# Try to import summarization functionality
try:
    from article_summarizer import summarize_article_url
    SUMMARIZATION_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Article summarization functionality loaded successfully")
except ImportError as e:
    SUMMARIZATION_AVAILABLE = False
    summarize_article_url = None
    logger = logging.getLogger(__name__)
    logger.warning(f"Article summarization not available: {e}")
except Exception as e:
    SUMMARIZATION_AVAILABLE = False
    summarize_article_url = None
    logger = logging.getLogger(__name__)
    logger.error(f"Error loading article summarization: {e}")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NewsAPI.org configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWSAPI_BASE_URL = 'https://newsapi.org/v2/'

# In-memory storage for articles
articles_cache = []
cache_timestamp = None

def categorize_article(title, description):
    """Categorize article based on title and description"""
    text = f"{title} {description}".lower()
    
    if any(word in text for word in ['sport', 'game', 'championship', 'team', 'player', 'match']):
        return 'sports'
    elif any(word in text for word in ['tech', 'ai', 'computer', 'software', 'digital', 'cyber']):
        return 'tech'
    elif any(word in text for word in ['business', 'economic', 'market', 'finance', 'stock', 'company']):
        return 'business'
    elif any(word in text for word in ['health', 'medical', 'disease', 'drug', 'hospital', 'doctor']):
        return 'health'
    elif any(word in text for word in ['entertainment', 'movie', 'music', 'celebrity', 'film', 'actor']):
        return 'entertainment'
    else:
        return 'politics'

def analyze_sentiment(text):
    """Enhanced sentiment analysis with better scoring and more comprehensive word lists"""
    # More comprehensive word lists with weights - specifically tuned for news content
    positive_words = {
        # Strong positive (weight 3)
        'excellent': 3, 'amazing': 3, 'fantastic': 3, 'outstanding': 3, 'brilliant': 3,
        'exceptional': 3, 'remarkable': 3, 'spectacular': 3, 'magnificent': 3, 'superb': 3,
        'triumph': 3, 'victory': 3, 'breakthrough': 3, 'revolutionary': 3, 'innovative': 3,
        
        # Moderate positive (weight 2)
        'great': 2, 'good': 2, 'wonderful': 2, 'success': 2, 'achievement': 2, 'accomplish': 2,
        'win': 2, 'progress': 2, 'improve': 2, 'benefit': 2, 'growth': 2, 'advance': 2,
        'effective': 2, 'efficient': 2, 'profitable': 2, 'successful': 2, 'thriving': 2,
        'flourishing': 2, 'prosperous': 2, 'booming': 2, 'rising': 2, 'soaring': 2,
        
        # Mild positive (weight 1)
        'positive': 1, 'rise': 1, 'gain': 1, 'boost': 1, 'better': 1, 'strong': 1,
        'healthy': 1, 'stable': 1, 'recovery': 1, 'solution': 1, 'hope': 1, 'optimistic': 1,
        'approve': 1, 'support': 1, 'launch': 1, 'expand': 1, 'increase': 1, 'up': 1,
        'celebrate': 1, 'praise': 1, 'commend': 1, 'welcome': 1, 'pleased': 1, 'satisfied': 1,
        'confident': 1, 'promising': 1, 'encouraging': 1, 'uplifting': 1, 'inspiring': 1
    }
    
    negative_words = {
        # Strong negative (weight 3)
        'terrible': 3, 'awful': 3, 'horrible': 3, 'disaster': 3, 'catastrophe': 3,
        'devastating': 3, 'tragic': 3, 'horrific': 3, 'appalling': 3, 'shocking': 3,
        'outrageous': 3, 'scandalous': 3, 'alarming': 3, 'disturbing': 3, 'terrifying': 3,
        
        # Moderate negative (weight 2)
        'crisis': 2, 'death': 2, 'killed': 2, 'murdered': 2, 'attack': 2, 'war': 2,
        'bad': 2, 'fail': 2, 'decline': 2, 'crash': 2, 'threat': 2, 'danger': 2,
        'violence': 2, 'conflict': 2, 'corruption': 2, 'fraud': 2, 'scandal': 2,
        'controversy': 2, 'protest': 2, 'riot': 2, 'strike': 2, 'recession': 2,
        
        # Mild negative (weight 1)
        'injured': 1, 'loss': 1, 'drop': 1, 'fall': 1, 'problem': 1, 'issue': 1,
        'concern': 1, 'worry': 1, 'risk': 1, 'damage': 1, 'hurt': 1, 'hit': 1,
        'controls': 1, 'restrictions': 1, 'shortfalls': 1, 'outbreaks': 1, 'spread': 1,
        'pressure': 1, 'firing': 1, 'tariffs': 1, 'unemployment': 1, 'inflation': 1,
        'criticism': 1, 'blame': 1, 'reject': 1, 'deny': 1, 'oppose': 1, 'struggle': 1,
        'difficulty': 1, 'challenge': 1, 'setback': 1, 'delay': 1, 'cancel': 1
    }
    
    # Context modifiers that can change sentiment intensity
    intensifiers = {
        'very': 1.5, 'extremely': 2.0, 'highly': 1.3, 'significantly': 1.4,
        'greatly': 1.4, 'substantially': 1.3, 'considerably': 1.3, 'remarkably': 1.5,
        'particularly': 1.2, 'especially': 1.2, 'notably': 1.2, 'increasingly': 1.3
    }
    
    diminishers = {
        'slightly': 0.7, 'somewhat': 0.8, 'relatively': 0.8, 'fairly': 0.8,
        'rather': 0.8, 'quite': 0.9, 'moderately': 0.8, 'partially': 0.7
    }
    
    negators = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor', 'without'}
    
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    # Calculate weighted sentiment scores with context awareness
    positive_score = 0
    negative_score = 0
    positive_words_found = []
    negative_words_found = []
    
    for i, word in enumerate(words):
        # Remove punctuation for better matching
        clean_word = word.strip('.,!?;:"()[]{}')
        
        # Check for negation in the previous 2 words
        negated = False
        for j in range(max(0, i-2), i):
            if words[j].strip('.,!?;:"()[]{}') in negators:
                negated = True
                break
        
        # Check for intensifiers/diminishers in the previous 2 words
        modifier = 1.0
        for j in range(max(0, i-2), i):
            prev_word = words[j].strip('.,!?;:"()[]{}')
            if prev_word in intensifiers:
                modifier = intensifiers[prev_word]
                break
            elif prev_word in diminishers:
                modifier = diminishers[prev_word]
                break
        
        # Calculate sentiment with context
        if clean_word in positive_words:
            base_score = positive_words[clean_word] * modifier
            if negated:
                negative_score += base_score  # Negated positive becomes negative
                negative_words_found.append(f"NOT {clean_word}")
            else:
                positive_score += base_score
                positive_words_found.append(clean_word)
                
        elif clean_word in negative_words:
            base_score = negative_words[clean_word] * modifier
            if negated:
                positive_score += base_score  # Negated negative becomes positive
                positive_words_found.append(f"NOT {clean_word}")
            else:
                negative_score += base_score
                negative_words_found.append(clean_word)
    
    # Debug logging
    logger.info(f"Sentiment analysis for: {text[:100]}...")
    logger.info(f"Positive words found: {positive_words_found} (score: {positive_score})")
    logger.info(f"Negative words found: {negative_words_found} (score: {negative_score})")
    
    # Determine sentiment label and score with improved scaling
    if positive_score > negative_score:
        label = 'positive'
        # Improved score calculation for positive sentiment
        intensity = positive_score - negative_score
        # Use logarithmic scaling for more realistic scores
        score = min(intensity / max(total_words, 1) * 8, 0.95)  # Scale down for realism
        score = max(score, 0.05)  # Minimum positive score
    elif negative_score > positive_score:
        label = 'negative'
        # Improved score calculation for negative sentiment
        intensity = negative_score - positive_score
        # Use logarithmic scaling for more realistic scores
        score = -min(intensity / max(total_words, 1) * 8, 0.95)  # Scale down for realism
        score = min(score, -0.05)  # Maximum negative score (closest to 0)
    else:
        label = 'neutral'
        score = 0.0
    
    # Enhanced confidence calculation
    total_sentiment_score = positive_score + negative_score
    if total_sentiment_score > 0:
        # Base confidence on sentiment word density and strength
        word_density = total_sentiment_score / max(total_words, 1)
        confidence = min(word_density * 3, 0.95)  # More conservative confidence
        confidence = max(confidence, 0.2)  # Minimum confidence when sentiment words are found
        
        # Boost confidence for strong sentiment words
        if total_sentiment_score >= 6:  # Strong sentiment indicators
            confidence = min(confidence * 1.2, 0.95)
    else:
        confidence = 0.05  # Very low confidence for neutral with no sentiment words
    
    result = {
        'score': round(score, 3),
        'label': label,
        'confidence': round(confidence, 3)
    }
    
    logger.info(f"Final sentiment result: {result}")
    return result

def calculate_reading_time(text):
    """Calculate reading time in minutes"""
    words = len(text.split())
    return max(1, round(words / 200))  # Average reading speed: 200 words per minute

def convert_to_ist(utc_date_str):
    """Convert UTC date string to IST"""
    if not utc_date_str:
        return datetime.now().isoformat()
    
    try:
        # Parse the UTC date
        utc_date = datetime.fromisoformat(utc_date_str.replace('Z', '+00:00'))
        # Convert to IST (UTC+5:30)
        ist_timezone = timezone(timedelta(hours=5, minutes=30))
        ist_date = utc_date.astimezone(ist_timezone)
        return ist_date.isoformat()
    except:
        return utc_date_str

def fetch_news_from_newsapi():
    """Fetch news from NewsAPI.org - top headlines"""
    try:
        logger.info("Fetching fresh news from NewsAPI.org...")
        
        # NewsAPI.org top headlines endpoint - Latest global news
        url = f"{NEWSAPI_BASE_URL}top-headlines"
        params = {
            'apiKey': NEWS_API_KEY,
            'pageSize': 20,
            'language': 'en',
            'sortBy': 'publishedAt'  # Get latest news first
        }
        
        response = requests.get(url, params=params, timeout=10)
        logger.info(f"NewsAPI.org response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            # Filter out articles with removed content
            valid_articles = []
            for article in articles:
                if (article.get('title') and 
                    article.get('description') and 
                    '[Removed]' not in article.get('title', '') and
                    '[Removed]' not in article.get('description', '')):
                    valid_articles.append(article)
            
            logger.info(f"Successfully fetched {len(valid_articles)} valid articles from NewsAPI.org")
            return valid_articles[:12]
        
        else:
            logger.error(f"NewsAPI.org error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching from NewsAPI.org: {e}")
        return []

def transform_article(api_article):
    """Transform NewsAPI.org article to our format - direct API response only"""
    if not api_article:
        logger.error("Received None or empty article")
        return None
        
    # Generate stable ID based on URL to ensure consistency across cache refreshes
    url = api_article.get('url')
    if url:
        # Create a stable ID based on URL hash
        article_id = hashlib.md5(url.encode()).hexdigest()
    else:
        # Fallback to UUID if no URL
        article_id = str(uuid.uuid4())
    
    title = api_article.get('title') or 'No Title'
    description = api_article.get('description') or ''
    raw_content = api_article.get('content') or ''
    
    # Use only NewsAPI.org data - no web scraping
    content_parts = []
    
    # 1. Add title as heading
    if title:
        content_parts.append(f"# {title}")
    
    # 2. Add description (API provides this)
    if description and description.strip():
        content_parts.append(description.strip())
    
    # 3. Add API content if available
    if raw_content and raw_content.strip():
        content_clean = raw_content.strip()
        # Only add if it's different from description
        if content_clean != description.strip():
            content_parts.append(content_clean)
    
    # 4. Add metadata
    metadata_parts = []
    
    # Author
    author = api_article.get('author')
    if author and author not in ['Unknown', None, '', 'N/A']:
        metadata_parts.append(f"**Author:** {author}")
    
    # Source
    source = api_article.get('source', {})
    source_name = source.get('name') if isinstance(source, dict) else str(source)
    if source_name and source_name != 'Unknown':
        metadata_parts.append(f"**Source:** {source_name}")
    
    # Publication date (convert to IST)
    pub_date = api_article.get('publishedAt')
    if pub_date:
        try:
            from datetime import timezone, timedelta
            parsed_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            ist_timezone = timezone(timedelta(hours=5, minutes=30))
            ist_date = parsed_date.astimezone(ist_timezone)
            formatted_date = ist_date.strftime('%B %d, %Y at %I:%M %p IST')
            metadata_parts.append(f"**Published:** {formatted_date}")
        except:
            metadata_parts.append(f"**Published:** {pub_date}")
    
    # 5. Add metadata if we have any
    if metadata_parts:
        content_parts.append('\n\n' + '\n\n'.join(metadata_parts))
    
    # 6. Add original article link
    if url:
        content_parts.append(f"\n**Read Full Article:** {url}")
    
    # Combine all content
    content = '\n\n'.join(content_parts) if content_parts else 'No content available'
    
    # Add simple notice
    content += f"""

---
**Note:** Content from NewsAPI.org. Full article available at the original source.
"""
    
    return {
        'id': article_id,
        'title': title,
        'summary': description[:200] + '...' if len(description) > 200 else description,
        'content': content,
        'author': api_article.get('author') or 'Unknown',
        'publishedAt': convert_to_ist(api_article.get('publishedAt')) or datetime.now().isoformat(),
        'url': api_article.get('url') or '',
        'imageUrl': api_article.get('urlToImage') or 'https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg',
        'category': categorize_article(title, description),
        'sentiment': analyze_sentiment(f"{title} {description}"),
        'readingTime': calculate_reading_time(content),
        'source': source_name or 'Unknown',
        'createdAt': datetime.now(),
        'aiSummary': None
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'News API is running'
    })

@app.route('/api/news', methods=['GET'])
def get_news():
    """Fetch news articles directly from NewsAPI.org"""
    global articles_cache, cache_timestamp
    
    try:
        # Check for force refresh parameter
        force_refresh = request.args.get('force_refresh', '').lower() == 'true'
        
        # Check cache validity (2 minutes for fresher news)
        if not force_refresh and cache_timestamp and articles_cache:
            time_diff = datetime.now() - cache_timestamp
            if time_diff.total_seconds() < 120:  # 2 minutes for fresher content
                logger.info(f"Returning cached articles ({len(articles_cache)} articles)")
                return jsonify({
                    'success': True,
                    'articles': articles_cache,
                    'count': len(articles_cache),
                    'source': 'cache',
                    'timestamp': cache_timestamp.isoformat(),
                    'cache_duration': '2 minutes'
                })
        
        # Fetch fresh articles
        logger.info("Fetching fresh articles from NewsAPI.org...")
        api_articles = fetch_news_from_newsapi()
        
        if not api_articles:
            logger.error("No articles returned from NewsAPI.org")
            return jsonify({
                'success': False,
                'error': 'No articles found',
                'articles': [],
                'count': 0
            }), 404
        
        # Transform articles
        articles = []
        for api_article in api_articles:
            try:
                transformed = transform_article(api_article)
                if transformed:
                    articles.append(transformed)
            except Exception as e:
                logger.error(f"Error transforming article: {e}")
                continue
        
        # Update cache
        articles_cache = articles
        cache_timestamp = datetime.now()
        
        logger.info(f"Successfully fetched {len(articles)} articles from NewsAPI.org")
        return jsonify({
            'success': True,
            'articles': articles,
            'count': len(articles),
            'source': 'newsapi_force_refresh' if force_refresh else 'newsapi',
            'timestamp': datetime.now().isoformat(),
            'cache_duration': '2 minutes'
        })
        
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch news: {str(e)}',
            'articles': [],
            'count': 0
        }), 500

@app.route('/api/news/fetch', methods=['POST'])
def fetch_and_store_news():
    """Fetch and store news articles from NewsAPI.org"""
    global articles_cache, cache_timestamp
    
    try:
        logger.info("Fetching and storing fresh articles from NewsAPI.org...")
        api_articles = fetch_news_from_newsapi()
        
        if not api_articles:
            logger.error("No articles returned from NewsAPI.org")
            return jsonify({
                'success': False,
                'error': 'No articles found',
                'articles': [],
                'count': 0
            }), 404
        
        # Process and store articles using existing transform function
        processed_articles = []
        for article in api_articles:
            transformed_article = transform_article(article)
            if transformed_article:
                processed_articles.append(transformed_article)
        
        # Update cache
        articles_cache = processed_articles
        cache_timestamp = datetime.now()
        
        logger.info(f"Successfully processed and cached {len(processed_articles)} articles")
        
        return jsonify({
            'success': True,
            'articles': processed_articles,
            'count': len(processed_articles),
            'source': 'fresh',
            'timestamp': cache_timestamp.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching and storing news: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch news: {str(e)}',
            'articles': [],
            'count': 0
        }), 500

@app.route('/api/news/<article_id>', methods=['GET'])
def get_article(article_id):
    """Get specific article by ID"""
    global articles_cache, cache_timestamp
    
    try:
        # Check cache for the article
        if articles_cache:
            article = next((a for a in articles_cache if a['id'] == article_id), None)
            if article:
                return jsonify({
                    'success': True,
                    'article': article,
                    'source': 'cache'
                })
        
        # If cache is empty or article not found, try to populate cache
        logger.info(f"Article {article_id} not found in cache, attempting to populate cache...")
        
        # Fetch fresh articles to populate cache
        api_articles = fetch_news_from_newsapi()
        
        if api_articles:
            # Process and store articles using existing transform function
            processed_articles = []
            for article in api_articles:
                transformed_article = transform_article(article)
                if transformed_article:
                    processed_articles.append(transformed_article)
            
            # Update cache
            articles_cache = processed_articles
            cache_timestamp = datetime.now()
            
            logger.info(f"Cache populated with {len(processed_articles)} articles")
            
            # Try to find the article again
            article = next((a for a in articles_cache if a['id'] == article_id), None)
            if article:
                return jsonify({
                    'success': True,
                    'article': article,
                    'source': 'fresh'
                })
        
        # Article still not found
        return jsonify({
            'success': False,
            'error': 'Article not found',
            'article': None
        }), 404
        
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch article: {str(e)}',
            'article': None
        }), 500

@app.route('/api/news/<article_id>/summarize', methods=['POST'])
def summarize_article_endpoint(article_id):
    """Generate AI summary for a specific article"""
    global articles_cache
    
    try:
        # Check if summarization is available
        if not SUMMARIZATION_AVAILABLE or summarize_article_url is None:
            return jsonify({
                'success': False,
                'error': 'Summarization functionality not available. Install required dependencies: pip install newspaper3k transformers',
                'summary': 'Unable to summarize this article.'
            })
        
        # Find the article in cache
        if not articles_cache:
            return jsonify({
                'success': False,
                'error': 'No articles in cache',
                'summary': 'Unable to summarize this article.'
            })
        
        article = next((a for a in articles_cache if a['id'] == article_id), None)
        if not article:
            return jsonify({
                'success': False,
                'error': 'Article not found',
                'summary': 'Unable to summarize this article.'
            })
        
        # Get the article URL
        article_url = article.get('url')
        if not article_url:
            return jsonify({
                'success': False,
                'error': 'Article URL not available',
                'summary': 'Unable to summarize this article.'
            })
        
        logger.info(f"Generating summary for article: {article_id}")
        logger.info(f"Article URL: {article_url}")
        
        # Generate summary using trafilatura + BART
        summary_result = summarize_article_url(article_url)
        
        if summary_result['success']:
            # Update the article in cache with the AI summary and analyzed sentiment
            for cached_article in articles_cache:
                if cached_article['id'] == article_id:
                    cached_article['aiSummary'] = summary_result['summary']
                    # Update sentiment if available
                    if 'sentiment' in summary_result:
                        cached_article['sentiment'] = summary_result['sentiment']
                    break
            
            logger.info(f"Successfully generated summary for article {article_id}")
            response_data = {
                'success': True,
                'summary': summary_result['summary'],
                'processing_time': summary_result['processing_time'],
                'content_length': summary_result.get('content_length', 0),
                'summary_length': summary_result.get('summary_length', 0)
            }
            
            # Include sentiment in response if available
            if 'sentiment' in summary_result:
                response_data['sentiment'] = summary_result['sentiment']
            
            return jsonify(response_data)
        else:
            logger.error(f"Failed to generate summary for article {article_id}: {summary_result.get('error', 'Unknown error')}")
            
            # Return a 200 status code with error information
            response_data = {
                'success': False,
                'error': summary_result.get('error', 'Summarization failed'),
                'summary': summary_result['summary'],  # This will be "Unable to summarize this article."
                'processing_time': summary_result['processing_time']
            }
            
            # Include paywall information if available
            if 'is_paywall' in summary_result:
                response_data['is_paywall'] = summary_result['is_paywall']
                
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error summarizing article {article_id}: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to summarize article: {str(e)}',
            'summary': 'Unable to summarize this article.'
        })

@app.route('/api/warmup', methods=['POST'])
def warmup_models():
    """Warmup AI models to reduce first-request latency"""
    try:
        if not SUMMARIZATION_AVAILABLE or summarize_article_url is None:
            return jsonify({
                'success': False,
                'error': 'Summarization functionality not available'
            }), 500
        
        logger.info("Warming up AI models...")
        
        # Get the summarizer instance to trigger model loading
        from article_summarizer import get_summarizer
        summarizer = get_summarizer()
        
        # Pre-load both models
        summarizer._load_summarizer()
        summarizer._load_sentiment_analyzer()
        
        return jsonify({
            'success': True,
            'message': 'AI models warmed up successfully',
            'summarizer_loaded': summarizer._summarizer_loaded,
            'sentiment_loaded': summarizer._sentiment_loaded
        })
        
    except Exception as e:
        logger.error(f"Error warming up models: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to warm up models: {str(e)}'
        }), 500

@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Get the current status of AI models"""
    try:
        if not SUMMARIZATION_AVAILABLE or summarize_article_url is None:
            return jsonify({
                'success': False,
                'summarization_available': False,
                'error': 'Summarization functionality not available'
            })
        
        from article_summarizer import get_summarizer
        summarizer = get_summarizer()
        
        return jsonify({
            'success': True,
            'summarization_available': True,
            'summarizer_loaded': summarizer._summarizer_loaded,
            'sentiment_loaded': summarizer._sentiment_loaded,
            'dependencies_available': summarizer.dependencies_available,
            'content_extraction_available': summarizer.content_extraction_available
        })
        
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to check model status: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("Starting News API server...")
    logger.info(f"Using NewsAPI.org with key: {NEWS_API_KEY[:10]}...")
    app.run(debug=True, host='0.0.0.0', port=5000)