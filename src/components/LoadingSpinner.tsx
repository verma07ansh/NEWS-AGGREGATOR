import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex items-center justify-center min-h-96">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-blue-200 dark:border-blue-800 rounded-full animate-spin"></div>
        <div className="absolute inset-0 w-16 h-16 border-t-4 border-blue-500 rounded-full animate-spin"></div>
      </div>
    </div>
  );
};

export default LoadingSpinner;