import React from 'react';
import { useNavigate } from 'react-router-dom';
import { NewsArticle } from '../types';
import { useNews } from '../context/NewsContext';
import { 
  Clock, 
  ExternalLink,
  User,
  Calendar
} from 'lucide-react';

interface NewsCardProps {
  article: NewsArticle;
}

const NewsCard: React.FC<NewsCardProps> = React.memo(({ article }) => {
  const navigate = useNavigate();
  const { markAsRead } = useNews();

  const handleCardClick = () => {
    markAsRead(article.id);
    navigate(`/article/${article.id}`);
  };

  const handleExternalClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    window.open(article.url, '_blank', 'noopener,noreferrer');
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div
      onClick={handleCardClick}
      className="group relative bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:transform hover:scale-105 cursor-pointer"
    >
      <div className="aspect-video rounded-t-xl overflow-hidden">
        <img
          src={article.imageUrl}
          alt={article.title}
          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
          onError={(e) => {
            e.currentTarget.src = 'https://images.pexels.com/photos/518543/pexels-photo-518543.jpeg?auto=compress&cs=tinysrgb&w=400&h=250&dpr=1';
          }}
        />
      </div>
      
      <div className="p-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
              {article.category}
            </span>
          </div>
          <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400">
            <Clock className="h-3 w-3" />
            <span className="text-xs">{article.readingTime} min read</span>
          </div>
        </div>
        
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 line-clamp-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
          {article.title}
        </h3>
        
        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
          {article.summary}
        </p>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-1">
              <User className="h-3 w-3" />
              <span className="truncate max-w-24">{article.author}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Calendar className="h-3 w-3" />
              <span>{formatDate(article.publishedAt)}</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handleExternalClick}
              className="p-1 text-gray-400 hover:text-blue-500 transition-colors"
              title="Open original article"
            >
              <ExternalLink className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

NewsCard.displayName = 'NewsCard';

export default NewsCard;