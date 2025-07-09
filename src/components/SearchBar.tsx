import React, { useState, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { useNews } from '../context/NewsContext';

const SearchBar: React.FC = () => {
  const { filters, setFilters } = useNews();
  const [searchTerm, setSearchTerm] = useState(filters.searchQuery);

  useEffect(() => {
    const delayedSearch = setTimeout(() => {
      setFilters({ ...filters, searchQuery: searchTerm });
    }, 300);

    return () => clearTimeout(delayedSearch);
  }, [searchTerm, filters, setFilters]);

  const clearSearch = () => {
    setSearchTerm('');
    setFilters({ ...filters, searchQuery: '' });
  };

  return (
    <div className="relative">
      <div className="relative flex items-center">
        <Search className="absolute left-3 h-4 w-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search articles..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-10 py-2 bg-white/20 dark:bg-gray-800/50 backdrop-blur-sm border border-gray-200 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 placeholder-gray-500 dark:placeholder-gray-400"
        />
        {searchTerm && (
          <button
            onClick={clearSearch}
            className="absolute right-3 p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
};

export default SearchBar;