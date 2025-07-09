import { NewsArticle, NewsCategory } from '../types';

export const generateMockArticles = (): NewsArticle[] => {
  const categories: NewsCategory[] = ['politics', 'sports', 'tech', 'business', 'entertainment', 'health'];
  const sentiments = ['positive', 'negative', 'neutral'] as const;
  
  const sampleArticles = [
    {
      title: "Revolutionary AI Breakthrough Changes Everything",
      summary: "Scientists have developed a new AI system that can predict market trends with 95% accuracy, potentially revolutionizing the financial sector.",
      content: "In a groundbreaking development, researchers at MIT have unveiled an artificial intelligence system that demonstrates unprecedented accuracy in predicting market movements. The system, dubbed 'MarketMind', uses advanced neural networks to analyze vast amounts of financial data, social media sentiment, and economic indicators to forecast market trends with remarkable precision.",
      author: "Dr. Sarah Chen",
      category: 'tech' as NewsCategory,
      source: "TechNews Daily"
    },
    {
      title: "Global Climate Summit Reaches Historic Agreement",
      summary: "World leaders unite on ambitious climate goals, setting new standards for carbon reduction and renewable energy adoption.",
      content: "The Global Climate Summit concluded with a historic agreement signed by 195 nations, committing to unprecedented climate action. The agreement includes binding targets for carbon neutrality by 2040 and a $500 billion fund for developing nations to transition to clean energy.",
      author: "Maria Rodriguez",
      category: 'politics' as NewsCategory,
      source: "Global News Network"
    },
    {
      title: "Quantum Computing Reaches Commercial Viability",
      summary: "Major tech companies announce the first commercially viable quantum computers, promising to solve complex problems in minutes.",
      content: "Google, IBM, and several startups have announced that quantum computing has finally reached commercial viability. These new systems can solve certain types of complex mathematical problems exponentially faster than traditional computers, opening up new possibilities in drug discovery, cryptography, and financial modeling.",
      author: "Dr. James Wilson",
      category: 'tech' as NewsCategory,
      source: "Quantum Today"
    },
    {
      title: "Championship Victory Sparks City-Wide Celebration",
      summary: "Local team wins national championship after 20-year drought, bringing joy to millions of fans across the region.",
      content: "The Metropolitan Eagles secured their first national championship in two decades with a stunning 28-21 victory over the defending champions. The win sparked massive celebrations throughout the city, with an estimated 2 million fans lining the streets for the victory parade.",
      author: "Mike Johnson",
      category: 'sports' as NewsCategory,
      source: "Sports Central"
    },
    {
      title: "New Medical Treatment Shows Promise for Rare Disease",
      summary: "Clinical trials reveal breakthrough treatment that could help thousands of patients with previously incurable conditions.",
      content: "A new gene therapy treatment has shown remarkable success in clinical trials, offering hope to patients with rare genetic disorders. The treatment, developed by a team at Johns Hopkins, has achieved a 90% success rate in reversing symptoms of the condition.",
      author: "Dr. Emily Foster",
      category: 'health' as NewsCategory,
      source: "Medical Journal"
    },
    {
      title: "Entertainment Industry Embraces Virtual Reality",
      summary: "Major studios announce plans to create immersive VR experiences, revolutionizing how we consume entertainment.",
      content: "Hollywood's biggest studios are investing heavily in virtual reality technology, with several major VR films and interactive experiences scheduled for release next year. This shift could fundamentally change how audiences engage with entertainment content.",
      author: "Lisa Park",
      category: 'entertainment' as NewsCategory,
      source: "Entertainment Weekly"
    },
    {
      title: "Economic Recovery Exceeds All Expectations",
      summary: "Latest data shows the economy growing at unprecedented rates, with unemployment at historic lows.",
      content: "The latest economic indicators show robust growth across all sectors, with GDP expanding by 4.2% in the last quarter. Unemployment has dropped to 3.1%, the lowest level in 50 years, while consumer confidence reaches new highs.",
      author: "Robert Davis",
      category: 'business' as NewsCategory,
      source: "Financial Times"
    },
    {
      title: "Space Tourism Takes Flight with First Commercial Mission",
      summary: "Private space company successfully launches first commercial space tourism flight, marking a new era in space travel.",
      content: "SpaceX's latest mission carried the first group of commercial space tourists to orbit, marking a significant milestone in the commercialization of space travel. The three-day mission included scientific experiments and spectacular views of Earth.",
      author: "Dr. Amanda Zhang",
      category: 'tech' as NewsCategory,
      source: "Space Today"
    }
  ];

  return sampleArticles.map((article, index) => ({
    id: `article-${index + 1}`,
    ...article,
    publishedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    url: `https://example.com/article-${index + 1}`,
    imageUrl: `https://images.pexels.com/photos/${2000000 + index}/pexels-photo-${2000000 + index}.jpeg?auto=compress&cs=tinysrgb&w=400&h=250&dpr=1`,
    sentiment: {
      score: Math.random() * 2 - 1,
      label: sentiments[Math.floor(Math.random() * sentiments.length)],
      confidence: 0.7 + Math.random() * 0.3
    },
    readingTime: Math.floor(Math.random() * 8) + 2
  }));
};

export const mockArticles = generateMockArticles();