import { UserPreferences } from '../types';

const STORAGE_KEY = 'news-aggregator-preferences';

export const defaultPreferences: UserPreferences = {
  categories: ['politics', 'sports', 'tech', 'business'],
  theme: 'dark',
  bookmarks: [],
  readingHistory: [],
  lastVisit: new Date().toISOString()
};

export const loadPreferences = (): UserPreferences => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...defaultPreferences, ...JSON.parse(stored) };
    }
  } catch (error) {
    console.error('Failed to load preferences:', error);
  }
  return defaultPreferences;
};

export const savePreferences = (preferences: UserPreferences): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  } catch (error) {
    console.error('Failed to save preferences:', error);
  }
};

export const addBookmark = (articleId: string): void => {
  const preferences = loadPreferences();
  if (!preferences.bookmarks.includes(articleId)) {
    preferences.bookmarks.push(articleId);
    savePreferences(preferences);
  }
};

export const removeBookmark = (articleId: string): void => {
  const preferences = loadPreferences();
  preferences.bookmarks = preferences.bookmarks.filter(id => id !== articleId);
  savePreferences(preferences);
};

export const addToReadingHistory = (articleId: string): void => {
  const preferences = loadPreferences();
  const history = preferences.readingHistory.filter(id => id !== articleId);
  history.unshift(articleId);
  preferences.readingHistory = history.slice(0, 50); // Keep only last 50 articles
  savePreferences(preferences);
};