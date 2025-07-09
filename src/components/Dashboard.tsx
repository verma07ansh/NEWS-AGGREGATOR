import React, { useEffect, useState, useMemo, useRef } from 'react';
import { useNews } from '../context/NewsContext';
import { newsAPI } from '../services/api';
import NewsCard from './NewsCard';
import CategoryFilter from './CategoryFilter';
import LoadingSpinner from './LoadingSpinner';
import { RefreshCw, Wifi, WifiOff, AlertCircle } from 'lucide-react';
import HorizontalFilters from './HorizontalFilters';

const Dashboard: React.FC = () => {
  const { filteredArticles, isLoading, hasLoadedInitially, setArticles, setIsLoading } = useNews();
  const [fetchingNews, setFetchingNews] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [error, setError] = useState<string | null>(null);
  const hasInitialized = useRef(false);

  useEffect(() => {
    if (!hasInitialized.current) {
      hasInitialized.current = true;
      checkConnection();
      // Only load news if we haven't loaded initially
      if (!hasLoadedInitially) {
        loadNews();
      }
    }
  }, []); // Remove dependency to prevent re-running

  const checkConnection = async () => {
    try {
      setConnectionStatus('checking');
      await newsAPI.healthCheck();
      setConnectionStatus('connected');
    } catch (error) {
      setConnectionStatus('disconnected');
      console.error('Backend connection failed:', error);
    }
  };

  const loadNews = async () => {
    try {
      // Only show loading if we don't have articles already
      if (!hasLoadedInitially) {
        setIsLoading(true);
      }
      setError(null);
      // Force refresh to ensure we get fresh data with new ID format
      const articles = await newsAPI.getNews(true);
      setArticles(articles);
    } catch (error) {
      console.error('Error loading news:', error);
      setError('Failed to load news articles. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchFreshNews = async () => {
    try {
      setFetchingNews(true);
      setError(null);
      console.log('Fetching fresh news...');
      
      // Clear session storage to force fresh data
      sessionStorage.removeItem('newsArticles');
      
      // Use force refresh for "Fetch Fresh News" button
      const articles = await newsAPI.getNews(true);
      console.log('Fresh articles received:', articles.length);
      
      setArticles(articles);
      
      // Re-check connection after successful fetch
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Error fetching fresh news:', error);
      setError('Failed to fetch fresh news. Please check your connection and try again.');
      setConnectionStatus('disconnected');
    } finally {
      setFetchingNews(false);
    }
  };

  // Memoize the articles to prevent unnecessary re-renders
  const memoizedArticles = useMemo(() => filteredArticles, [filteredArticles]);

  // Only show loading spinner on true initial load
  if (isLoading && !hasLoadedInitially && memoizedArticles.length === 0) {
    return <LoadingSpinner />;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Status Bar */}
      <div className="mb-6 space-y-4 sm:space-y-0 sm:flex sm:items-center sm:justify-between">
        <div className="flex items-center justify-center sm:justify-start space-x-4">
          <div className="flex items-center space-x-3">
            {connectionStatus === 'connected' ? (
              <Wifi className="h-6 w-6 text-green-500" />
            ) : connectionStatus === 'disconnected' ? (
              <WifiOff className="h-6 w-6 text-red-500" />
            ) : (
              <RefreshCw className="h-6 w-6 text-yellow-500 animate-spin" />
            )}
            <span className="text-base font-medium text-gray-700 dark:text-gray-300">
              Backend: {connectionStatus}
            </span>
          </div>
          
          {error && (
            <div className="flex items-center space-x-2 text-red-600 dark:text-red-400 mt-2 sm:mt-0">
              <AlertCircle className="h-5 w-5" />
              <span className="text-sm font-medium">{error}</span>
            </div>
          )}
        </div>

        <div className="flex justify-center sm:justify-end">
          <button
            onClick={fetchFreshNews}
            disabled={fetchingNews || connectionStatus !== 'connected'}
            className="flex items-center space-x-3 px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl font-medium"
          >
            <RefreshCw className={`h-5 w-5 ${fetchingNews ? 'animate-spin' : ''}`} />
            <span className="text-base">{fetchingNews ? 'Fetching...' : 'Fetch Fresh News'}</span>
          </button>
        </div>
      </div>

      {/* Horizontal Filters for Mobile/Tablet */}
      <div className="lg:hidden mb-6">
        <HorizontalFilters />
      </div>

      <div className="lg:grid lg:grid-cols-4 lg:gap-8">
        {/* Desktop Filters Sidebar */}
        <div className="hidden lg:block lg:col-span-1">
          <div className="sticky top-24 space-y-6">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-xl p-6 shadow-lg">
              <CategoryFilter />
            </div>
          </div>
        </div>

        {/* News Grid */}
        <div className="lg:col-span-3">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Latest News
            </h2>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {memoizedArticles.length} articles found
            </div>
          </div>

          {memoizedArticles.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-gray-400 dark:text-gray-500 text-lg">
                {connectionStatus === 'disconnected' 
                  ? 'Backend server is not available'
                  : 'No articles found matching your criteria'
                }
              </div>
              <p className="text-gray-500 dark:text-gray-400 mt-2">
                {connectionStatus === 'disconnected'
                  ? 'Please make sure the Python backend is running'
                  : 'Try fetching fresh news or adjusting your filters'
                }
              </p>
              {connectionStatus === 'disconnected' && (
                <button
                  onClick={checkConnection}
                  className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  Retry Connection
                </button>
              )}
            </div>
          ) : (
            <div className="grid gap-8 md:grid-cols-2 xl:grid-cols-3">
              {memoizedArticles.map((article) => (
                <NewsCard key={article.id} article={article} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;