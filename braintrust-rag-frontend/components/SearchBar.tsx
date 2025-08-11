'use client';

import { useState, FormEvent } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading: boolean;
  hasConversation?: boolean;
}

export default function SearchBar({ onSearch, loading, hasConversation }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
      setQuery(''); // Clear the input after sending
    }
  };

  const exampleQueries = hasConversation ? [
    "Can you explain that in more detail?",
    "What are the best practices for this?", 
    "Can you show me a code example?",
    "How does this compare to other approaches?"
  ] : [
    "How do I set up API authentication?",
    "What's new in the latest release?",
    "How to use the Python SDK?",
    "Tracing and logging examples"
  ];

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none z-10">
            <MagnifyingGlassIcon className="h-4 w-4 text-gray-400 group-focus-within:text-gray-600 transition-colors" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={hasConversation ? "Ask a follow-up question..." : "Ask me anything about Braintrust APIs, SDKs, features..."}
            className="block w-full pl-11 pr-20 py-3 text-sm text-gray-900 border border-gray-300 rounded-md focus:ring-1 focus:ring-black focus:border-black bg-white transition-all placeholder:text-gray-500"
            disabled={loading}
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-2">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="inline-flex items-center px-3 py-1.5 border border-transparent text-sm font-medium rounded-md text-white bg-black hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-1 h-3 w-3 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Search
                </>
              ) : (
                <>
                  <MagnifyingGlassIcon className="h-3 w-3 mr-1" />
                  Search
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Example queries */}
      <div className="mt-6">
        <p className="text-sm font-medium text-gray-600 mb-3">
          {hasConversation ? "Continue the conversation:" : "Popular questions:"}
        </p>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuery(example)}
              className="inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium bg-gray-50 border border-gray-200 text-gray-700 hover:bg-gray-100 hover:border-gray-300 transition-all"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}