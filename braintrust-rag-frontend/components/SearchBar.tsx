'use client';

import { useState, FormEvent } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading: boolean;
}

export default function SearchBar({ onSearch, loading }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const exampleQueries = [
    "How do I set up API authentication?",
    "What's new in the latest release?",
    "How to use the Python SDK?",
    "Tracing and logging examples"
  ];

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none z-10">
            <MagnifyingGlassIcon className="h-5 w-5 text-gray-500 group-focus-within:text-indigo-500 transition-colors" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask me anything about Braintrust APIs, SDKs, features..."
            className="block w-full pl-14 pr-36 py-4 text-base text-gray-900 border-2 border-gray-300 rounded-2xl shadow-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white transition-all duration-200 placeholder:text-gray-500 placeholder:font-normal font-medium"
            disabled={loading}
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-3">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="inline-flex items-center px-6 py-2.5 border border-transparent text-sm font-semibold rounded-xl text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl disabled:hover:shadow-lg"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Searching...
                </>
              ) : (
                <>
                  <MagnifyingGlassIcon className="h-4 w-4 mr-1" />
                  Search
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Example queries */}
      <div className="mt-8">
        <p className="text-sm font-medium text-gray-700 mb-4">Popular questions:</p>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuery(example)}
              className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-white border border-gray-200 text-gray-700 hover:bg-indigo-50 hover:border-indigo-300 hover:text-indigo-700 transition-all duration-200 shadow-sm hover:shadow-md"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}