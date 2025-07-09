import axios from 'axios';
import { NewsArticle, SummarizedArticle } from '../types';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // Increased to 2 minutes for AI model loading
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const newsAPI = {
  // Fetch and store news from external API
  fetchAndStoreNews: async (): Promise<NewsArticle[]> => {
    try {
      const response = await api.post('/news/fetch');
      return response.data.articles || [];
    } catch (error) {
      console.error('Error fetching and storing news:', error);
      throw error;
    }
  },

  // Get all news articles from database
  getNews: async (forceRefresh = false): Promise<NewsArticle[]> => {
    try {
      const url = forceRefresh ? '/news?force_refresh=true' : '/news';
      const response = await api.get(url);
      const articles = response.data.articles || [];
      console.log('API getNews received articles:', articles.length);
      if (articles.length > 0) {
        console.log('First article sentiment from API:', articles[0].sentiment);
      }
      return articles;
    } catch (error) {
      console.error('Error getting news:', error);
      throw error;
    }
  },

  // Get specific article by ID
  getArticle: async (articleId: string): Promise<NewsArticle> => {
    try {
      const response = await api.get(`/news/${articleId}`);
      return response.data.article;
    } catch (error) {
      console.error('Error getting article:', error);
      throw error;
    }
  },

  // Summarize article using AI
  summarizeArticle: async (articleId: string): Promise<any> => {
    try {
      const response = await api.post(`/news/${articleId}/summarize`, {
        length: 'detailed', // Request a longer, more detailed summary
        max_sentences: 8,   // Request up to 8 sentences instead of default
        include_key_points: true, // Include key points analysis
        style: 'comprehensive' // Request comprehensive analysis
      });
      
      // Return the full response data which may include sentiment
      if (response.data.sentiment) {
        return {
          summary: response.data.summary,
          sentiment: response.data.sentiment,
          processing_time: response.data.processing_time,
          content_length: response.data.content_length,
          summary_length: response.data.summary_length
        };
      }
      
      // Fallback to just the summary for backward compatibility
      return response.data.summary;
    } catch (error) {
      console.error('Error summarizing article:', error);
      throw error;
    }
  },

  // Health check
  healthCheck: async (): Promise<any> => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Warmup AI models
  warmupModels: async (): Promise<any> => {
    try {
      const response = await api.post('/warmup');
      return response.data;
    } catch (error) {
      console.error('Model warmup failed:', error);
      throw error;
    }
  },

  // Get model status
  getModelsStatus: async (): Promise<any> => {
    try {
      const response = await api.get('/models/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get model status:', error);
      throw error;
    }
  }
};

export default api;