import React from 'react';
import { NewsCategory } from '../types';
import { useNews } from '../context/NewsContext';
import { 
  Globe, 
  Trophy, 
  Laptop, 
  TrendingUp, 
  Film, 
  Heart,
  Grid3X3
} from 'lucide-react';

const HorizontalFilters: React.FC = () => {
  const { filters, setFilters } = useNews();

  const categories = [
    { id: 'all', label: 'All', icon: Grid3X3 },
    { id: 'politics', label: 'Politics', icon: Globe },
    { id: 'sports', label: 'Sports', icon: Trophy },
    { id: 'tech', label: 'Tech', icon: Laptop },
    { id: 'business', label: 'Business', icon: TrendingUp },
    { id: 'entertainment', label: 'Entertainment', icon: Film },
    { id: 'health', label: 'Health', icon: Heart },
  ];


  const handleCategoryChange = (categoryId: string) => {
    setFilters({ ...filters, category: categoryId as NewsCategory | 'all' });
  };

  const handleSentimentChange = (sentimentId: string) => {
    setFilters({ ...filters, sentiment: sentimentId as 'all' | 'positive' | 'negative' | 'neutral' });
  };

  return (
    <div className="space-y-4">
      {/* Categories */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 px-1">Categories</h3>
        <div className="flex space-x-2 overflow-x-auto pb-2 scrollbar-hide">
          {categories.map((category) => {
            const IconComponent = category.icon;
            const isActive = filters.category === category.id;
            
            return (
              <button
                key={category.id}
                onClick={() => handleCategoryChange(category.id)}
                className={`flex items-center space-x-2 px-4 py-2.5 rounded-full whitespace-nowrap transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-500 text-white shadow-lg'
                    : 'bg-white/80 dark:bg-gray-800/80 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 shadow-md'
                } backdrop-blur-sm`}
              >
                <IconComponent className="h-4 w-4" />
                <span className="text-sm font-medium">{category.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default HorizontalFilters;