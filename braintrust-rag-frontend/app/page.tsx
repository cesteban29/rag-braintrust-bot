'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import SearchResults from '@/components/SearchResults';
import ConversationDisplay from '@/components/ConversationDisplay';
import { Source, Message, QueryResponse } from '@/types';

export default function Home() {
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState('');
  const [answer, setAnswer] = useState<string>('');
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    setCurrentQuery(query);
    
    // Immediately add the user's query to conversation history for initial questions
    const isNewConversation = conversationHistory.length === 0;
    if (isNewConversation) {
      setConversationHistory([{ role: 'user', content: query }]);
    } else {
      // For follow-ups, add the user query immediately
      setConversationHistory(prev => [...prev, { role: 'user', content: query }]);
    }
    
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query, 
          top_k: 5,
          conversation_history: conversationHistory,
          conversation_id: conversationId
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data: QueryResponse = await response.json();
      setSources(data.sources);
      setAnswer(data.answer);
      setConversationHistory(data.conversation_history);
      setConversationId(data.conversation_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setSources([]);
      setAnswer('');
      // If there was an error, revert the conversation history
      if (isNewConversation) {
        setConversationHistory([]);
      } else {
        setConversationHistory(prev => prev.slice(0, -1));
      }
    } finally {
      setLoading(false);
    }
  };

  const clearConversation = () => {
    setConversationHistory([]);
    setSources([]);
    setAnswer('');
    setCurrentQuery('');
    setError(null);
    setConversationId(null); // Clear the conversation ID to start a new trace
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-8 h-8 bg-black rounded-md flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2L2 7v10c0 5.55 3.84 10 9 11 1.09.2 2.09.2 3 0 5.16-1 9-5.45 9-11V7l-10-5z"/>
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Braintrust Docs Assistant
                </h1>
                <p className="text-sm text-gray-500">
                  AI-powered assistant for documentation
                </p>
              </div>
            </div>
            {conversationHistory.length > 0 && (
              <button
                onClick={clearConversation}
                className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 hover:text-gray-900 bg-gray-50 border border-gray-200 rounded-md hover:bg-gray-100 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Search Section - Only show when no conversation */}
        {conversationHistory.length === 0 && (
          <div className="max-w-4xl mx-auto mb-8">
            <SearchBar 
              onSearch={handleSearch} 
              loading={loading} 
              hasConversation={false}
            />
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="max-w-4xl mx-auto mb-8">
            <div className="bg-red-50 border border-red-200 p-4 rounded-md">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-800 font-medium">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Conversation Display */}
        {conversationHistory.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-md p-6 mb-8">
            <ConversationDisplay 
              messages={conversationHistory} 
              loading={loading} 
              onFollowUp={handleSearch}
            />
          </div>
        )}

        {/* Sources */}
        {sources.length > 0 && !loading && (
          <div className="mt-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Source Documents</h3>
            <SearchResults query={currentQuery} sources={sources} />
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="max-w-4xl mx-auto text-center py-12">
            <div className="inline-flex items-center px-4 py-2 border border-gray-200 text-sm font-medium rounded-md text-gray-700 bg-gray-50">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Searching documentation...
            </div>
          </div>
        )}

        {/* Empty State */}
        {conversationHistory.length === 0 && !loading && (
          <div className="text-center py-16">
            <div className="max-w-lg mx-auto">
              <div className="mx-auto h-12 w-12 bg-gray-50 rounded-md flex items-center justify-center mb-4">
                <svg className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Start a conversation
              </h3>
              <p className="text-gray-600 mb-6">
                Ask me anything about Braintrust APIs, SDKs, features, or documentation. I'll provide detailed answers with relevant source material.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}