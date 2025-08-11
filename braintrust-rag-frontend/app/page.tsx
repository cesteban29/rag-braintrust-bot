'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import SearchResults from '@/components/SearchResults';
import { Source } from '@/types';

export default function Home() {
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState('');

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    setCurrentQuery(query);
    
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setSources(data.sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setSources([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-100 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
              Braintrust Docs Search
            </h1>
            <p className="text-gray-600">
              AI-powered semantic search through Braintrust documentation
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Search Section */}
        <div className="max-w-3xl mx-auto mb-8">
          <SearchBar onSearch={handleSearch} loading={loading} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-3xl mx-auto mb-8">
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-r-lg">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700 font-medium">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {currentQuery && !loading && !error && (
          <SearchResults query={currentQuery} sources={sources} />
        )}

        {/* Loading State */}
        {loading && (
          <div className="max-w-3xl mx-auto text-center py-12">
            <div className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-indigo-100">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Searching through documentation...
            </div>
          </div>
        )}

        {/* Empty State */}
        {!currentQuery && !loading && (
          <div className="text-center py-16">
            <div className="max-w-md mx-auto">
              <svg className="mx-auto h-16 w-16 text-indigo-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Ready to search
              </h3>
              <p className="text-gray-500 mb-6">
                Ask me anything about Braintrust APIs, SDKs, features, or documentation
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}