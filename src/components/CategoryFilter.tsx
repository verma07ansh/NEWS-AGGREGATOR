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

const CategoryFilter: React.FC = () => {
  const { filters, setFilters } = useNews();

  const categories = [
    { id: 'all', label: 'All News', icon: Grid3X3 },
    { id: 'politics', label: 'Politics', icon: Globe },
    { id: 'sports', label: 'Sports', icon: Trophy },
    { id: 'tech', label: 'Technology', icon: Laptop },
    { id: 'business', label: 'Business', icon: TrendingUp },
    { id: 'entertainment', label: 'Entertainment', icon: Film },
    { id: 'health', label: 'Health', icon: Heart },
  ];

  const handleCategoryChange = (categoryId: string) => {
    setFilters({ ...filters, category: categoryId as NewsCategory | 'all' });
  };

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Categories</h3>
      <div className="space-y-1">
        {categories.map((category) => {
          const IconComponent = category.icon;
          const isActive = filters.category === category.id;
          
          return (
            <button
              key={category.id}
              onClick={() => handleCategoryChange(category.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-all duration-200 ${
                isActive
                  ? 'bg-blue-500 text-white shadow-lg transform scale-105'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              <IconComponent className="h-4 w-4" />
              <span className="font-medium">{category.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default CategoryFilter;