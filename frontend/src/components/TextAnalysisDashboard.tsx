import React, { useState, useEffect } from 'react';
import { 
  BookOpen, 
  LineChart, 
  Text, 
  Upload,
  BarChart2,
  Layers,
  Compass 
} from 'lucide-react';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5002';

interface AnalysisResults {
  summary: {
    total_sentences: number;
    total_words: number;
    unique_words: number;
    reading_level: string;
    named_entities: Array<[string, string]>;
  };
  pos_analysis: Array<[string, number]>;
  sentiment: {
    nltk_sentiments: {
      neg: number;
      neu: number;
      pos: number;
      compound: number;
    }[];
    avg_sentiment_score: number;
    spacy_sentiment_vectors: number[];
  };
  tfidf_keywords: Array<[string, number]>;
  visualizations: {
    word_frequency: string;
    pos_distribution: string;
  };
}

const AdvancedTextAnalysisDashboard = () => {
  const [availableTexts, setAvailableTexts] = useState<string[]>([]);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('summary');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // Fetch available texts on mount
  useEffect(() => {
    const fetchTexts = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/texts`);
        const data = await response.json();
        setAvailableTexts(data);
        if (data.length) setSelectedText(data[0]);
      } catch (err) {
        setError('Failed to fetch texts');
        console.error(err);
      }
    };
    fetchTexts();
  }, []);

  // Perform text analysis
  const performAnalysis = async () => {
    if (!selectedText) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze/${selectedText}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze text');
      }
      
      setAnalysisResults(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // File upload handler
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (!uploadedFile) return;

    const formData = new FormData();
    formData.append('file', uploadedFile);

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setAnalysisResults(data);
      setSelectedText(uploadedFile.name);
      setIsUploadModalOpen(false);
    } catch (err) {
      setError('File upload failed');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  if (error) {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-semibold mb-2">Error</h3>
          <p className="text-red-600">{error}</p>
          <button 
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 max-w-5xl mx-auto">
      <div className="bg-white shadow-md rounded-lg overflow-hidden">
        <div className="px-6 py-4 bg-gray-50 flex items-center justify-between">
          <div className="flex items-center">
            <BookOpen className="mr-2 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-800">
              Advanced Text Analysis
            </h2>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="relative">
              <select 
                value={selectedText || ''}
                onChange={(e) => setSelectedText(e.target.value)}
                className="block w-64 px-3 py-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                aria-label="Select text for analysis"
              >
                {availableTexts.map((text) => (
                  <option key={text} value={text}>
                    {text}
                  </option>
                ))}
              </select>
            </div>

            <button 
              onClick={() => setIsUploadModalOpen(true)}
              className="p-2 bg-gray-200 rounded-md hover:bg-gray-300 transition-colors"
              aria-label="Upload text file"
              title="Upload text file"
            >
              <Upload className="text-gray-600" />
            </button>
          </div>
        </div>
        
        {/* Analysis Button */}
        <div className="px-6 py-4">
          <button 
            onClick={performAnalysis} 
            disabled={isLoading || !selectedText}
            className={`w-full py-2 rounded-md transition-colors ${
              isLoading || !selectedText 
                ? 'bg-gray-300 cursor-not-allowed' 
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>
        
        {/* Analysis Results */}
        {analysisResults && (
          <div className="p-6">
            {/* Tabs */}
            <div className="flex border-b mb-4">
              {[
                { id: 'summary', label: 'Summary', icon: Text },
                { id: 'pos', label: 'POS', icon: Layers },
                { id: 'sentiment', label: 'Sentiment', icon: Compass },
                { id: 'keywords', label: 'Keywords', icon: BarChart2 },
                { id: 'visuals', label: 'Visualizations', icon: LineChart }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-4 py-2 border-b-2 ${
                    activeTab === tab.id 
                      ? 'border-blue-600 text-blue-600' 
                      : 'border-transparent text-gray-500'
                  }`}
                >
                  <tab.icon className="mr-2 w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div>
              {activeTab === 'summary' && (
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-gray-100 p-4 rounded">
                    <h3 className="font-bold mb-2">Text Statistics</h3>
                    <p>Total Sentences: {analysisResults.summary.total_sentences}</p>
                    <p>Total Words: {analysisResults.summary.total_words}</p>
                    <p>Unique Words: {analysisResults.summary.unique_words}</p>
                    <p>Reading Level: {analysisResults.summary.reading_level}</p>
                  </div>
                  <div className="bg-gray-100 p-4 rounded">
                    <h3 className="font-bold mb-2">Named Entities</h3>
                    <ul>
                      {analysisResults.summary.named_entities.map((entity, index) => (
                        <li key={index}>{entity[0]} ({entity[1]})</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {/* Additional tab contents can be added similarly */}
            </div>
          </div>
        )}
      </div>

      {/* File Upload Modal */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl">
            <h2 className="text-xl font-semibold mb-4">Upload Text File</h2>
            <input 
              type="file" 
              accept=".txt"
              onChange={handleFileUpload}
              aria-label="Choose text file to upload"
              title="Choose text file to upload"
              className="w-full"
            />
            <div className="mt-4 flex justify-end space-x-2">
              <button 
                onClick={() => setIsUploadModalOpen(false)}
                className="px-4 py-2 bg-gray-200 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedTextAnalysisDashboard;