import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { NewsArticle, FilterOptions, UserPreferences } from '../types';
import { loadPreferences, savePreferences } from '../utils/localStorage';

interface NewsContextType {
  articles: NewsArticle[];
  filteredArticles: NewsArticle[];
  filters: FilterOptions;
  preferences: UserPreferences;
  isLoading: boolean;
  hasLoadedInitially: boolean;
  setArticles: (articles: NewsArticle[]) => void;
  setFilters: (filters: FilterOptions) => void;
  setIsLoading: (loading: boolean) => void;
  toggleBookmark: (articleId: string) => void;
  markAsRead: (articleId: string) => void;
  updatePreferences: (preferences: UserPreferences) => void;
  updateArticleSentiment: (articleId: string, sentiment: any) => void;
}

const NewsContext = createContext<NewsContextType | undefined>(undefined);

export const useNews = () => {
  const context = useContext(NewsContext);
  if (!context) {
    throw new Error('useNews must be used within a NewsProvider');
  }
  return context;
};

export const NewsProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Load articles from sessionStorage if available
  const loadArticlesFromSession = (): NewsArticle[] => {
    try {
      const stored = sessionStorage.getItem('newsArticles');
      if (stored) {
        const articles = JSON.parse(stored);
        // Check if articles have old UUID format IDs (36 chars with dashes)
        // New format uses MD5 hashes (32 chars, no dashes)
        if (articles.length > 0 && articles[0].id && articles[0].id.length === 36 && articles[0].id.includes('-')) {
          console.log('Detected old article ID format, clearing cache...');
          sessionStorage.removeItem('newsArticles');
          return [];
        }
        return articles;
      }
      return [];
    } catch {
      return [];
    }
  };

  const [articles, setArticles] = useState<NewsArticle[]>(loadArticlesFromSession());
  const [filteredArticles, setFilteredArticles] = useState<NewsArticle[]>([]);
  const [filters, setFilters] = useState<FilterOptions>({
    category: 'all',
    sentiment: 'all',
    searchQuery: ''
  });
  const [preferences, setPreferences] = useState<UserPreferences>(loadPreferences());
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoadedInitially, setHasLoadedInitially] = useState(loadArticlesFromSession().length > 0);

  useEffect(() => {
    let filtered = articles;

    // Filter by category
    if (filters.category !== 'all') {
      filtered = filtered.filter(article => article.category === filters.category);
    }

    // Filter by sentiment
    if (filters.sentiment !== 'all') {
      filtered = filtered.filter(article => article.sentiment.label === filters.sentiment);
    }

    // Filter by search query
    if (filters.searchQuery) {
      const query = filters.searchQuery.toLowerCase();
      filtered = filtered.filter(article =>
        article.title.toLowerCase().includes(query) ||
        article.summary.toLowerCase().includes(query) ||
        article.content.toLowerCase().includes(query)
      );
    }

    setFilteredArticles(filtered);
  }, [articles, filters]);

  const toggleBookmark = (articleId: string) => {
    const newBookmarks = preferences.bookmarks.includes(articleId)
      ? preferences.bookmarks.filter(id => id !== articleId)
      : [...preferences.bookmarks, articleId];

    const newPreferences = { ...preferences, bookmarks: newBookmarks };
    setPreferences(newPreferences);
    savePreferences(newPreferences);
  };

  const markAsRead = (articleId: string) => {
    const newHistory = preferences.readingHistory.filter(id => id !== articleId);
    newHistory.unshift(articleId);
    const newPreferences = { ...preferences, readingHistory: newHistory.slice(0, 50) };
    setPreferences(newPreferences);
    savePreferences(newPreferences);
  };

  const updatePreferences = (newPreferences: UserPreferences) => {
    setPreferences(newPreferences);
    savePreferences(newPreferences);
  };

  const updateArticleSentiment = (articleId: string, sentiment: any) => {
    setArticles(prevArticles => {
      const updatedArticles = prevArticles.map(article => 
        article.id === articleId 
          ? { ...article, sentiment }
          : article
      );
      
      // Update sessionStorage as well
      try {
        sessionStorage.setItem('newsArticles', JSON.stringify(updatedArticles));
      } catch (error) {
        console.warn('Failed to update articles in sessionStorage:', error);
      }
      
      return updatedArticles;
    });
  };

  const handleSetArticles = (newArticles: NewsArticle[]) => {
    console.log('Setting new articles in context:', newArticles.length);
    console.log('First article sentiment:', newArticles[0]?.sentiment);
    setArticles(newArticles);
    // Save to sessionStorage for persistence across navigation
    try {
      sessionStorage.setItem('newsArticles', JSON.stringify(newArticles));
    } catch (error) {
      console.warn('Failed to save articles to sessionStorage:', error);
    }
    if (newArticles.length > 0) {
      setHasLoadedInitially(true);
    }
  };

  const handleSetIsLoading = (loading: boolean) => {
    // Never show loading if we have articles in memory or cache
    const cachedArticles = loadArticlesFromSession();
    const hasArticles = articles.length > 0 || cachedArticles.length > 0;
    
    if (loading === false) {
      // Always allow setting loading to false
      setIsLoading(false);
    } else if (!hasArticles && !hasLoadedInitially) {
      // Only show loading if we truly have no articles and haven't loaded before
      setIsLoading(true);
    }
    // Otherwise, ignore the loading request to prevent fade-out
  };

  return (
    <NewsContext.Provider value={{
      articles,
      filteredArticles,
      filters,
      preferences,
      isLoading,
      hasLoadedInitially,
      setArticles: handleSetArticles,
      setFilters,
      setIsLoading: handleSetIsLoading,
      toggleBookmark,
      markAsRead,
      updatePreferences,
      updateArticleSentiment
    }}>
      {children}
    </NewsContext.Provider>
  );
};