# AI News Aggregator with Smart Summarization

A modern news aggregation platform that fetches articles from multiple sources and provides AI-powered summarization with hallucination prevention.

## 🚀 Features

- **Multi-Source News Aggregation**: Fetches articles from various news APIs
- **AI-Powered Summarization**: Uses BART model for intelligent article summarization
- **Hallucination Prevention**: Advanced validation to ensure summaries contain only factual information from source articles
- **Sentiment Analysis**: Analyzes the sentiment of articles
- **Modern UI**: Clean, responsive interface built with React and TypeScript
- **Real-time Updates**: Live news feed with automatic refresh
- **Category Filtering**: Filter news by categories
- **Search Functionality**: Search through articles
- **Dark/Light Theme**: Toggle between themes

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Context API** for state management

### Backend
- **Flask** (Python) REST API
- **BART Model** (`facebook/bart-large-cnn`) for summarization
- **Transformers** library for AI processing
- **BeautifulSoup** for content extraction
- **Newspaper3k** for article parsing

## 📁 Project Structure

```
project/
├── src/                    # Frontend React application
│   ├── components/         # React components
│   ├── context/           # Context providers
│   ├── services/          # API services
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
├── backend/               # Flask backend
│   ├── app.py            # Main Flask application
│   ├── article_summarizer.py  # AI summarization logic
│   ├── requirements.txt   # Python dependencies
│   ├── test_*.py         # Test files
│   └── .env.example      # Environment variables template
├── package.json          # Frontend dependencies
└── README.md            # This file
```

## 🔧 Installation & Setup

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **News API Key** from [NewsAPI.org](https://newsapi.org/)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env    # macOS/Linux
   ```
   
   Edit `.env` and add your News API key:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

5. **Run the backend server:**
   ```bash
   python app.py
   ```
   
   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to project root:**
   ```bash
   cd ..  # if you're in backend directory
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:5173`

## 🧪 Testing

### Backend Tests

The backend includes comprehensive tests for the summarization functionality:

```bash
cd backend

# Test hallucination fix
python test_hallucination_fix.py

# Test additional edge cases
python test_additional_cases.py
```

### Test Results
- ✅ Hallucination prevention working
- ✅ Extractive fallback functional
- ✅ Entity preservation accurate
- ✅ Multiple article types supported

## 🤖 AI Summarization Features

### Hallucination Prevention
The system includes advanced measures to prevent AI hallucination:

- **Text Preprocessing**: Cleans input to reduce model confusion
- **Entity Validation**: Compares numbers, names, and dates between source and summary
- **Extractive Fallback**: Uses rule-based summarization when AI fails
- **Conservative Parameters**: Optimized model settings to reduce hallucination

### Example Fix
**Before (Hallucinated):**
> "At least 94 people, including 28 children, have been confirmed dead following the flash floods..."

**After (Accurate):**
> "More than a dozen summer camps dot the banks of the Guadalupe River. Many camps are adjacent to high-risk flood zones according to FEMA maps."

## 📚 API Endpoints

### News Endpoints
- `GET /api/news` - Fetch latest news articles
- `GET /api/news?category=technology` - Filter by category
- `GET /api/news?q=search_term` - Search articles

### Summarization Endpoints
- `POST /api/news/{article_id}/summarize` - Generate AI summary for article

### Response Format
```json
{
  "success": true,
  "summary": "Article summary text...",
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.85
  },
  "content_length": 1500,
  "summary_length": 150,
  "processing_time": 2.3
}
```

## 🔒 Environment Variables

Create a `.env` file in the backend directory:

```env
# Required
NEWS_API_KEY=your_newsapi_key_here

# Optional
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000
```

## 🚀 Deployment

### Backend Deployment
1. Set environment variables on your hosting platform
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`

### Frontend Deployment
1. Build the project: `npm run build`
2. Deploy the `dist` folder to your hosting platform

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'transformers'"**
   ```bash
   pip install transformers torch
   ```

2. **"BART model loading failed"**
   - Check internet connection (model downloads on first use)
   - Ensure sufficient disk space (~1.6GB for BART model)

3. **"News API rate limit exceeded"**
   - Check your API key limits at NewsAPI.org
   - Implement caching for development

4. **Frontend can't connect to backend**
   - Ensure backend is running on port 5000
   - Check CORS settings in Flask app

## 📈 Performance

- **Model Loading**: ~10-15 seconds on first startup
- **Summarization**: ~2-5 seconds per article
- **Memory Usage**: ~2GB RAM for BART model
- **Fallback**: <1 second for extractive summarization
