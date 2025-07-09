export interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  author: string;
  publishedAt: string;
  url: string;
  imageUrl: string;
  category: NewsCategory;
  sentiment: SentimentScore;
  readingTime: number;
  source: string;
  description?: string;
  urlToImage?: string;
}

export interface SentimentScore {
  score: number; // -1 to 1
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

export type NewsCategory = 'politics' | 'sports' | 'tech' | 'business' | 'entertainment' | 'health';

export interface UserPreferences {
  categories: NewsCategory[];
  theme: 'light' | 'dark';
  bookmarks: string[];
  readingHistory: string[];
  lastVisit: string;
}

export interface FilterOptions {
  category: NewsCategory | 'all';
  sentiment: 'all' | 'positive' | 'negative' | 'neutral';
  searchQuery: string;
}

export interface NewsAPIResponse {
  status: string;
  totalResults: number;
  articles: NewsAPIArticle[];
}

export interface NewsAPIArticle {
  source: {
    id: string | null;
    name: string;
  };
  author: string | null;
  title: string;
  description: string | null;
  url: string;
  urlToImage: string | null;
  publishedAt: string;
  content: string | null;
}

export interface SummarizedArticle extends NewsArticle {
  aiSummary?: string;
  isLoading?: boolean;
}