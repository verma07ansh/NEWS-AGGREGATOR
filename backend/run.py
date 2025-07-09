#!/usr/bin/env python3
"""
News Aggregator Backend Server
Run this script to start the Flask backend server
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set environment variables if not already set
    if not os.getenv('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'development'
    
    if not os.getenv('FLASK_DEBUG'):
        os.environ['FLASK_DEBUG'] = 'True'
    
    print("🚀 Starting News Aggregator Backend Server...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - GET  /api/health - Health check")
    print("   - GET  /api/news - Get all news articles")
    print("   - POST /api/news/fetch - Fetch and store fresh news")
    print("   - GET  /api/news/<id> - Get specific article")
    print("   - POST /api/news/<id>/summarize - Summarize article")
    print("\n💡 Make sure to:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Set up MongoDB Atlas cluster and get connection string")
    print("   3. Add your MongoDB Atlas URI and News API key to .env file")
    print("   4. Start the React frontend: npm run dev")
    print("\n🔗 MongoDB Atlas Setup:")
    print("   1. Create account at https://www.mongodb.com/atlas")
    print("   2. Create a free cluster")
    print("   3. Create database user and whitelist IP")
    print("   4. Get connection string and update .env file")
    print("\n" + "="*50)
    
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)