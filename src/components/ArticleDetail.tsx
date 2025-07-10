import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  Clock, 
  User, 
  Calendar, 
  ExternalLink, 
  Sparkles,
  Loader2,
  Globe,
  Zap,
  Star,
  Users,
  Lightbulb,
  Activity,
  RefreshCw,
  Lock
} from 'lucide-react';
import { NewsArticle, SummarizedArticle } from '../types';
import { newsAPI } from '../services/api';
import { useNews } from '../context/NewsContext';

const ArticleDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { preferences, updateArticleSentiment } = useNews();
  
  const [article, setArticle] = useState<SummarizedArticle | null>(null);
  const [loading, setLoading] = useState(true);
  const [summarizing, setSummarizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readingProgress, setReadingProgress] = useState(0);

  useEffect(() => {
    if (id) {
      loadArticle(id);
    }
  }, [id]);

  // Reading progress tracking
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (scrollTop / docHeight) * 100;
      setReadingProgress(Math.min(progress, 100));
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);



  const loadArticle = async (articleId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      // First try to get fresh news to ensure we have updated sentiment data
      await newsAPI.getNews(true);
      
      // Then get the specific article
      const articleData = await newsAPI.getArticle(articleId);
      setArticle(articleData);
    } catch (error) {
      console.error('Error loading article:', error);
      setError('Failed to load article. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!article) return;

    try {
      setSummarizing(true);
      setError(null);
      const response = await newsAPI.summarizeArticle(article.id);
      
      // Check if the response indicates an error
      if (response && typeof response === 'object' && response.success === false) {
        // Handle error response - store the error message but don't clear aiSummary
        setError(response.error || 'Failed to generate summary');
        return;
      }
      
      // Handle both string response (old format) and object response (new format)
      if (typeof response === 'string') {
        setArticle(prev => prev ? { ...prev, aiSummary: response } : null);
        setError(null); // Clear any previous errors
      } else if (response && typeof response === 'object') {
        // Check if there's a note about the summary source
        if (response.note) {
          setError(`Note: ${response.note}`);
        } else {
          setError(null); // Clear any previous errors
        }
        
        setArticle(prev => {
          if (!prev) return null;
          const updated = { ...prev, aiSummary: response.summary || response };
          
          // Update sentiment if available in the response
          if (response.sentiment) {
            updated.sentiment = response.sentiment;
          }
          
          return updated;
        });
      }
    } catch (error) {
      console.error('Error summarizing article:', error);
      
      // Extract more detailed error information if available
      let errorMessage = 'Failed to generate summary. Please try again.';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Summary generation is taking longer than expected. The AI models may be loading for the first time. Please try again in a moment.';
      } else if (error.response && error.response.data) {
        // Extract error message from API response
        const responseData = error.response.data;
        if (responseData.error) {
          errorMessage = responseData.error;
        }
      }
      
      setError(errorMessage);
    } finally {
      setSummarizing(false);
    }
  };



  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'Asia/Kolkata'
    });
  };



  const getReadingTimeColor = (time: number) => {
    if (time <= 3) return 'text-green-600 dark:text-green-400';
    if (time <= 7) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const formatArticleContent = (content: string) => {
    // Split content into paragraphs and format them
    const paragraphs = content.split('\n').filter(p => p.trim().length > 0);
    
    return (
      <div className="space-y-4 sm:space-y-6">
        {paragraphs.map((paragraph, index) => {
          // Check if it's a title/header (usually the first line or lines with specific patterns)
          if (index === 0 && paragraph.includes(' - ')) {
            const [title, source] = paragraph.split(' - ');
            return (
              <div key={index} className="mb-6 sm:mb-8">
                <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4 leading-tight">
                  {title.trim()}
                </h1>
                <div className="flex items-center space-x-2 text-xs sm:text-sm text-gray-600 dark:text-gray-400">
                  <Globe className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="font-medium">{source}</span>
                </div>
              </div>
            );
          }
          
          // Check if it's author/metadata info
          if (paragraph.includes('**Author:**') || paragraph.includes('**Source:**') || paragraph.includes('**Published:**')) {
            // Parse the metadata line to extract author, source, and published info
            const authorMatch = paragraph.match(/\*\*Author:\*\*\s*([^*]+)/);
            const sourceMatch = paragraph.match(/\*\*Source:\*\*\s*([^*]+)/);
            const publishedMatch = paragraph.match(/\*\*Published:\*\*\s*([^*]+)/);
            
            return (
              <div key={index} className="metadata-card mb-4 sm:mb-6">
                <div className="flex items-center space-x-2 mb-3">
                  <div className="p-1 bg-blue-500 rounded-full">
                    <Activity className="h-3 w-3 text-white" />
                  </div>
                  <span className="text-xs sm:text-sm font-semibold text-blue-600 dark:text-blue-400">Article Information</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 text-sm">
                  {authorMatch && (
                    <div className="flex items-center space-x-2 bg-white/50 dark:bg-gray-700/50 rounded-lg p-2">
                      <User className="h-4 w-4 text-blue-500" />
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Author</p>
                        <p className="font-medium text-gray-700 dark:text-gray-300">{authorMatch[1].trim()}</p>
                      </div>
                    </div>
                  )}
                  {sourceMatch && (
                    <div className="flex items-center space-x-2 bg-white/50 dark:bg-gray-700/50 rounded-lg p-2">
                      <Globe className="h-4 w-4 text-green-500" />
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Source</p>
                        <p className="font-medium text-gray-700 dark:text-gray-300">{sourceMatch[1].trim()}</p>
                      </div>
                    </div>
                  )}
                  {publishedMatch && (
                    <div className="flex items-center space-x-2 bg-white/50 dark:bg-gray-700/50 rounded-lg p-2">
                      <Calendar className="h-4 w-4 text-purple-500" />
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Published At</p>
                        <p className="font-medium text-gray-700 dark:text-gray-300">{publishedMatch[1].trim()}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          }
          
          // Check if it's a "Read Full Article" link
          if (paragraph.includes('**Read Full Article:**')) {
            const url = paragraph.replace('**Read Full Article:**', '').trim();
            return (
              <div key={index} className="link-card">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg shadow-sm">
                      <ExternalLink className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                    </div>
                    <div>
                      <p className="font-semibold text-gray-800 dark:text-gray-200 text-sm sm:text-base">Read Full Article</p>
                      <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Continue reading on the original source</p>
                    </div>
                  </div>
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3 sm:px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 text-xs sm:text-sm font-medium shadow-sm hover:shadow-md text-center w-full sm:w-auto"
                  >
                    Open Link
                  </a>
                </div>
              </div>
            );
          }
          
          // Check if it's a note or disclaimer
          if (paragraph.includes('**Note:**')) {
            const note = paragraph.replace('**Note:**', '').trim();
            return (
              <div key={index} className="note-card">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full mt-1 shadow-sm">
                    <Lightbulb className="h-4 w-4 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-2 mb-2 space-y-1 sm:space-y-0">
                      <p className="font-semibold text-yellow-800 dark:text-yellow-200 text-xs sm:text-sm">Important Note</p>
                      <div className="hidden sm:block h-1 w-1 bg-yellow-600 rounded-full"></div>
                      <p className="text-xs text-yellow-600 dark:text-yellow-400">Please read carefully</p>
                    </div>
                    <p className="text-yellow-700 dark:text-yellow-300 text-xs sm:text-sm leading-relaxed">{note}</p>
                  </div>
                </div>
              </div>
            );
          }
          
          // Regular paragraph
          const isFirstContentParagraph = index === 1 || (index === 0 && !paragraph.includes(' - '));
          return (
            <div key={index} className="mb-4 sm:mb-6">
              <p className={`text-gray-700 dark:text-gray-300 leading-relaxed text-base sm:text-lg ${
                isFirstContentParagraph 
                  ? 'first-paragraph text-justify' 
                  : 'text-justify'
              }`}>
                {paragraph.trim()}
              </p>
              {isFirstContentParagraph && (
                <div className="mt-2 flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
                  <div className="h-1 w-6 sm:w-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
                  <span className="text-xs sm:text-sm">Article content begins</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const formatSummaryContent = (summary: string) => {
    // Handle summary content that might contain structured text
    const sentences = summary.split('. ').filter(s => s.trim().length > 0);
    
    return (
      <div className="space-y-3">
        {sentences.map((sentence, index) => {
          const trimmedSentence = sentence.trim();
          if (!trimmedSentence) return null;
          
          // Add period back if it was removed by split
          const formattedSentence = trimmedSentence.endsWith('.') ? trimmedSentence : trimmedSentence + '.';
          
          return (
            <p key={index} className="text-gray-700 dark:text-gray-300 leading-relaxed">
              {index === 0 && (
                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400 mr-1">
                  {formattedSentence.charAt(0)}
                </span>
              )}
              {index === 0 ? formattedSentence.slice(1) : formattedSentence}
            </p>
          );
        })}
      </div>
    );
  };

  const getSentimentColor = (sentiment: any) => {
    if (!sentiment) return 'text-gray-500';
    
    if (sentiment.label === 'positive') {
      return 'text-green-600 dark:text-green-400';
    } else if (sentiment.label === 'negative') {
      return 'text-red-600 dark:text-red-400';
    }
    return 'text-gray-600 dark:text-gray-400';
  };

  const getSentimentIcon = (sentiment: any) => {
    if (!sentiment) return '😐';
    
    const score = sentiment.score || 0;
    if (sentiment.label === 'positive') {
      return score > 0.7 ? '😊' : score > 0.3 ? '🙂' : '😐';
    } else if (sentiment.label === 'negative') {
      return score < -0.7 ? '😞' : score < -0.3 ? '😕' : '😐';
    }
    return '😐';
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">Loading article...</p>
        </div>
      </div>
    );
  }

  // Only show the error page if there's no article data
  if (!article) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-4">Article not found</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Reading Progress Bar */}
      <div className="fixed top-0 left-0 w-full h-1 bg-gray-200 dark:bg-gray-700 z-50">
        <div 
          className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-300 ease-out"
          style={{ width: `${readingProgress}%` }}
        />
      </div>



      {/* Navigation Bar */}
      <div className="bg-white dark:bg-gray-900 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
          <button
            onClick={() => navigate('/')}
            className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="font-medium text-sm sm:text-base">Back to Dashboard</span>
          </button>
        </div>
      </div>

      {/* Article Header */}
      <div className="bg-white dark:bg-gray-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
          {/* Category and Meta Info */}
          <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 mb-4 sm:mb-6">
            <span className="inline-flex items-center px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-semibold bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 w-fit">
              {article.category}
            </span>
            <span className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
              {formatDate(article.publishedAt)}
            </span>
          </div>
          
          {/* Article Title */}
          <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-white mb-4 sm:mb-6 leading-tight">
            {article.title}
          </h1>
          
          {/* Author and Reading Info */}
          <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-3 sm:gap-6 text-gray-600 dark:text-gray-400 mb-6 sm:mb-8">
            <div className="flex items-center space-x-2">
              <User className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="font-medium text-sm sm:text-base">By {article.author}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Clock className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="text-sm sm:text-base">{article.readingTime} min read</span>
            </div>
            <div className="flex items-center space-x-2">
              <Globe className="h-3 w-3 sm:h-4 sm:w-4" />
              <span className="text-sm sm:text-base">{article.source}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Featured Image */}
      <div className="bg-white dark:bg-gray-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6">
          <div className="relative mb-6 sm:mb-8">
            <img
              src={article.imageUrl}
              alt={article.title}
              className="w-full h-48 sm:h-64 md:h-80 lg:h-96 object-cover rounded-lg shadow-lg"
              onError={(e) => {
                e.currentTarget.src = 'https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg?auto=compress&cs=tinysrgb&w=1200&h=600&dpr=1';
              }}
            />
            {/* Image Caption */}
            <div className="mt-2 sm:mt-3 text-xs sm:text-sm text-gray-500 dark:text-gray-400 text-center italic px-2">
              Featured image for: {article.title}
            </div>
          </div>
        </div>
      </div>



      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">

        {/* AI Summary Section */}
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg border border-purple-200 dark:border-purple-700 mb-6 sm:mb-8 hover:shadow-xl transition-all duration-300 pulse-glow">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 space-y-4 sm:space-y-0">
            <div className="flex items-center space-x-3">
              <div className="p-2 sm:p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl shadow-lg animate-pulse">
                <Zap className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">AI-Powered Summary</h2>
                <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Enhanced with artificial intelligence</p>
              </div>
            </div>
            
            {!article.aiSummary && (
              <div className="flex flex-col space-y-3">
                <button
                  onClick={handleSummarize}
                  disabled={summarizing}
                  className="px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl hover:-translate-y-0.5 text-sm sm:text-base w-full sm:w-auto"
                >
                  {summarizing ? (
                    <>
                      <Loader2 className="h-4 w-4 sm:h-5 sm:w-5 animate-spin" />
                      <span className="hidden sm:inline">Generating Detailed Summary...</span>
                      <span className="sm:hidden">Generating...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 sm:h-5 sm:w-5" />
                      <span className="hidden sm:inline">Generate Detailed AI Summary</span>
                      <span className="sm:hidden">Generate Summary</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
          
          {article.aiSummary ? (
            <div className="bg-white/70 dark:bg-gray-800/70 rounded-xl p-4 sm:p-6 border-l-4 border-purple-500 backdrop-blur-sm">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 space-y-2 sm:space-y-0">
                <div className="flex items-center space-x-2">
                  <Star className="h-4 w-4 text-purple-500" />
                  <span className="text-xs sm:text-sm font-medium text-purple-600 dark:text-purple-400">AI Generated Summary</span>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500 dark:text-gray-400">Analyzed Sentiment:</span>
                    <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                      article.sentiment?.label === 'positive' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : article.sentiment?.label === 'negative'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                    }`}>
                      {article.sentiment?.label ? 
                        article.sentiment.label.charAt(0).toUpperCase() + article.sentiment.label.slice(1) : 
                        'Neutral'
                      }
                    </span>
                  </div>
                  <button
                    onClick={handleSummarize}
                    disabled={summarizing}
                    className="flex items-center justify-center space-x-1 px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-300 shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
                    title="Regenerate AI Summary"
                  >
                    {summarizing ? (
                      <>
                        <Loader2 className="h-3 w-3 animate-spin" />
                        <span>Regenerating...</span>
                      </>
                    ) : (
                      <>
                        <RefreshCw className="h-3 w-3" />
                        <span>Regenerate Summary</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
              <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-base sm:text-lg font-medium">
                {article.aiSummary}
              </p>
              {/* Enhanced Sentiment Analysis for AI Summary */}
              <div className="mt-4 sm:mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3 space-y-2 sm:space-y-0">
                  <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">AI-Enhanced Sentiment Analysis</h3>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{getSentimentIcon(article.sentiment)}</span>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                      article.sentiment?.label === 'positive' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : article.sentiment?.label === 'negative'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                    }`}>
                      {article.sentiment?.label?.toUpperCase() || 'NEUTRAL'}
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 text-sm">
                  <div className="flex items-center space-x-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200 dark:border-purple-700">
                    <Activity className="h-4 w-4 text-purple-500" />
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Sentiment Intensity</p>
                      <p className="font-semibold text-gray-700 dark:text-gray-300">
                        {article.sentiment?.score !== undefined && article.sentiment.score !== null ? 
                          `${article.sentiment.score >= 0 ? '+' : ''}${(article.sentiment.score * 100).toFixed(1)}%` : 
                          '0.0%'
                        }
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200 dark:border-purple-700">
                    <Star className="h-4 w-4 text-yellow-500" />
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">AI Confidence</p>
                      <p className="font-semibold text-gray-700 dark:text-gray-300">
                        {article.sentiment?.confidence !== undefined && article.sentiment.confidence !== null ? 
                          `${(article.sentiment.confidence * 100).toFixed(1)}%` : 
                          '0.0%'
                        }
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 border border-purple-200 dark:border-purple-700">
                    <Zap className="h-4 w-4 text-indigo-500" />
                    <div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">Analysis Method</p>
                      <p className="font-semibold text-gray-700 dark:text-gray-300">
                        AI + Rules
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 italic">
                  * Sentiment analysis performed on full article content using advanced AI models
                </div>
              </div>
            </div>
          ) : error ? (
            <div className={`bg-white/70 dark:bg-gray-800/70 rounded-xl p-4 sm:p-6 border-l-4 ${
              error.includes("paywall") 
                ? "border-red-500" 
                : "border-yellow-500"
            } backdrop-blur-sm`}>
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 space-y-2 sm:space-y-0">
                <div className="flex items-center space-x-2">
                  {error.includes("paywall") ? (
                    <>
                      <Lock className="h-4 w-4 text-red-500" />
                      <span className="text-xs sm:text-sm font-medium text-red-600 dark:text-red-400">Paywall Detected</span>
                    </>
                  ) : (
                    <>
                      <Lightbulb className="h-4 w-4 text-yellow-500" />
                      <span className="text-xs sm:text-sm font-medium text-yellow-600 dark:text-yellow-400">Content Extraction Issue</span>
                    </>
                  )}
                </div>
                <button
                  onClick={handleSummarize}
                  disabled={summarizing}
                  className="flex items-center justify-center space-x-1 px-3 py-1.5 text-xs font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-300 shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
                  title="Try Again"
                >
                  {summarizing ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-3 w-3" />
                      <span>Try Again</span>
                    </>
                  )}
                </button>
              </div>
              
              <div className={`p-4 rounded-lg ${
                error.includes("paywall") 
                  ? "bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800" 
                  : "bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800"
              }`}>
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    {error.includes("paywall") ? (
                      <div className="p-1 bg-red-100 dark:bg-red-800 rounded-full">
                        <Lock className="h-5 w-5 text-red-500 dark:text-red-300" />
                      </div>
                    ) : (
                      <div className="p-1 bg-yellow-100 dark:bg-yellow-800 rounded-full">
                        <Lightbulb className="h-5 w-5 text-yellow-500 dark:text-yellow-300" />
                      </div>
                    )}
                  </div>
                  <div className="ml-3 flex-1">
                    <h3 className={`text-sm font-medium ${
                      error.includes("paywall") 
                        ? "text-red-800 dark:text-red-300" 
                        : "text-yellow-800 dark:text-yellow-300"
                    }`}>
                      {error.includes("paywall") ? "Paywall Detected" : "Content Extraction Issue"}
                    </h3>
                    <div className={`mt-1 text-sm ${
                      error.includes("paywall") 
                        ? "text-red-700 dark:text-red-200" 
                        : "text-yellow-700 dark:text-yellow-200"
                    }`}>
                      <p>{error}</p>
                      {error.includes("paywall") && (
                        <p className="mt-2">Try visiting the article directly to see subscription options.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6 sm:p-8 text-center border-2 border-dashed border-purple-300 dark:border-purple-600">
              <div className="flex flex-col items-center space-y-3">
                <div className="p-3 sm:p-4 bg-purple-100 dark:bg-purple-900/30 rounded-full">
                  <Sparkles className="h-6 w-6 sm:h-8 sm:w-8 text-purple-500" />
                </div>
                <p className="text-gray-600 dark:text-gray-400 text-base sm:text-lg">
                  Generate a detailed AI summary
                </p>
                <p className="text-gray-500 dark:text-gray-500 text-xs sm:text-sm px-4">
                  Click the button above to create a comprehensive, AI-powered analysis with key points and insights
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Full Article Content */}
        <div className="bg-gradient-to-br from-white to-green-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg border border-gray-100 dark:border-gray-700 mb-6 sm:mb-8 hover:shadow-xl transition-all duration-300">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 space-y-3 sm:space-y-0">
            <div className="flex items-center space-x-3">
              <div className="p-2 sm:p-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-xl shadow-lg">
                <Globe className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">Full Article</h2>
                <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Complete story with all details</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-xs sm:text-sm text-gray-500 dark:text-gray-400">
              <Users className="h-3 w-3 sm:h-4 sm:w-4" />
              <span>{article.readingTime} min read</span>
            </div>
          </div>
          
          <div className="prose prose-sm sm:prose-base lg:prose-lg dark:prose-invert max-w-none">
            <div className="text-gray-700 dark:text-gray-300 leading-relaxed text-base sm:text-lg font-serif">
              {formatArticleContent(article.content)}
            </div>
          </div>
        </div>

        {/* Quick Summary by NewsAPI.org */}
        <div className="bg-gradient-to-br from-white to-blue-50 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg border border-gray-100 dark:border-gray-700 mb-6 sm:mb-8 hover:shadow-xl transition-all duration-300 fade-in-scale">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6 space-y-3 sm:space-y-0">
            <div className="flex items-center space-x-3">
              <div className="p-2 sm:p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Lightbulb className="h-5 w-5 sm:h-6 sm:w-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">Quick Summary</h2>
                <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">Provided by NewsAPI.org</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-xs sm:text-sm text-gray-500 dark:text-gray-400">
              <Clock className="h-3 w-3 sm:h-4 sm:w-4" />
              <span>30 sec read</span>
            </div>
          </div>
          <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-4 sm:p-6 border-l-4 border-blue-500">
            <div className="text-base sm:text-lg text-gray-700 dark:text-gray-300 leading-relaxed font-medium">
              {formatSummaryContent(article.summary)}
            </div>
          </div>
        </div>


      </div>
    </div>
  );
};

export default ArticleDetail;