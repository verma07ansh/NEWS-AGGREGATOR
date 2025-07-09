import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { NewsProvider } from './context/NewsContext';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ArticleDetail from './components/ArticleDetail';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  return (
    <ThemeProvider>
      <NewsProvider>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-purple-900 transition-all duration-300">
            <Header />
            <main>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/article/:id" element={<ArticleDetail />} />
                <Route path="/loading" element={<LoadingSpinner />} />
              </Routes>
            </main>
          </div>
        </Router>
      </NewsProvider>
    </ThemeProvider>
  );
}

export default App;