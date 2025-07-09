import React from 'react';
import { Newspaper, Settings, User } from 'lucide-react';
import SearchBar from './SearchBar';

const Header: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-blue-500 rounded-lg">
                <Newspaper className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">NewsAI</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Smart News Aggregator</p>
              </div>
            </div>
          </div>
          
          <div className="flex-1 max-w-md mx-8">
            <SearchBar />
          </div>
          
          <div className="flex items-center space-x-4">
            {/* <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors">
              <Settings className="h-5 w-5" />
            </button>
            <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors">
              <User className="h-5 w-5" />
            </button> */}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;