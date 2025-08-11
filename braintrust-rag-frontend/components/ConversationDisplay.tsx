'use client';

import { Message } from '@/types';
import { UserIcon, SparklesIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';
import { useState, FormEvent } from 'react';

interface ConversationDisplayProps {
  messages: Message[];
  loading?: boolean;
  onFollowUp?: (query: string) => void;
}

export default function ConversationDisplay({ messages, loading = false, onFollowUp }: ConversationDisplayProps) {
  const [followUpQuery, setFollowUpQuery] = useState('');

  if (messages.length === 0 && !loading) {
    return null;
  }

  const handleFollowUpSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (followUpQuery.trim() && onFollowUp) {
      onFollowUp(followUpQuery.trim());
      setFollowUpQuery('');
    }
  };

  const quickFollowUps = [
    "Can you explain this in more detail?",
    "Show me a code example",
    "What are the best practices?",
    "How does this compare to alternatives?"
  ];

  return (
    <div className="space-y-6 mb-8">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Conversation</h2>
        <div className="text-sm text-gray-500 mb-4">
          {Math.floor(messages.length / 2)} exchange{Math.floor(messages.length / 2) !== 1 ? 's' : ''}
        </div>
      </div>
      
      <div className="space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${
              message.role === 'user' ? 'flex-row' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div className={`flex-shrink-0 w-8 h-8 rounded-md flex items-center justify-center ${
              message.role === 'user' 
                ? 'bg-gray-100 text-gray-600' 
                : 'bg-black text-white'
            }`}>
              {message.role === 'user' ? (
                <UserIcon className="h-4 w-4" />
              ) : (
                <SparklesIcon className="h-4 w-4" />
              )}
            </div>

            {/* Message Content */}
            <div className={`flex-1 min-w-0 ${
              message.role === 'user' ? 'max-w-3xl' : 'max-w-4xl'
            }`}>
              <div className={`rounded-md px-4 py-3 ${
                message.role === 'user'
                  ? 'bg-gray-50 border border-gray-200'
                  : 'bg-white border border-gray-200'
              }`}>
                <div className={`text-xs font-medium mb-2 ${
                  message.role === 'user' ? 'text-gray-600' : 'text-gray-600'
                }`}>
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div className={`prose prose-sm max-w-none ${
                  message.role === 'user' 
                    ? 'text-gray-800' 
                    : 'text-gray-800'
                }`}>
                  {message.role === 'assistant' ? (
                    <div 
                      className="prose prose-sm max-w-none leading-relaxed"
                      dangerouslySetInnerHTML={{
                        __html: message.content
                          // Handle code blocks first (```language\ncode\n```)
                          .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre class="bg-gray-900 text-gray-100 rounded-lg p-4 my-4 overflow-x-auto"><code class="font-mono text-sm">$2</code></pre>')
                          // Handle single backticks after code blocks
                          .replace(/`([^`\n]+)`/g, '<code class="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-gray-800">$1</code>')
                          // Handle headers
                          .replace(/^### (.*$)/gm, '<h4 class="text-base font-semibold text-gray-900 mt-4 mb-2">$1</h4>')
                          .replace(/^## (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-900 mt-5 mb-3">$1</h3>')
                          .replace(/^# (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mt-6 mb-3">$1</h2>')
                          // Handle bold and italic
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
                          .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
                          // Handle links
                          .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-indigo-600 hover:text-indigo-800 underline">$1</a>')
                          // Handle line breaks and paragraphs
                          .replace(/\n\n/g, '</p><p class="mb-3">')
                          // Handle bullet points
                          .replace(/^- (.*$)/gm, '<li class="ml-4">$1</li>')
                          .replace(/(<li.*<\/li>)/s, '<ul class="list-disc list-inside mb-3 space-y-1">$1</ul>')
                          // Handle numbered lists
                          .replace(/^\d+\. (.*$)/gm, '<li class="ml-4">$1</li>')
                          .replace(/(<li.*<\/li>)/s, '<ol class="list-decimal list-inside mb-3 space-y-1">$1</ol>')
                          // Wrap in paragraphs
                          .replace(/^(?!<[huo]|<pre|<li)(.+$)/gm, '<p class="mb-3">$1</p>')
                          // Clean up empty paragraphs
                          .replace(/<p[^>]*><\/p>/g, '')
                      }}
                    />
                  ) : (
                    <p className="leading-relaxed">{message.content}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {/* Loading indicator */}
        {loading && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-md bg-black text-white flex items-center justify-center">
              <SparklesIcon className="h-4 w-4" />
            </div>
            <div className="flex-1 min-w-0 max-w-4xl">
              <div className="rounded-md px-4 py-3 bg-white border border-gray-200">
                <div className="text-xs font-medium mb-2 text-gray-600">
                  Assistant
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-sm text-gray-500">Thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Follow-up Input Bar - Only show after conversation and when not loading */}
      {messages.length > 0 && !loading && onFollowUp && (
        <div className="mt-6 pt-6 border-t border-gray-200">
          <div className="flex gap-3 items-start">
            <div className="flex-shrink-0 w-8 h-8 rounded-md bg-gray-100 text-gray-600 flex items-center justify-center">
              <UserIcon className="h-4 w-4" />
            </div>
            <div className="flex-1 min-w-0">
              <form onSubmit={handleFollowUpSubmit} className="space-y-3">
                <div className="relative">
                  <input
                    type="text"
                    value={followUpQuery}
                    onChange={(e) => setFollowUpQuery(e.target.value)}
                    placeholder="Ask a follow-up question..."
                    className="block w-full pl-3 pr-10 py-2 text-sm text-gray-900 border border-gray-300 rounded-md focus:ring-1 focus:ring-black focus:border-black bg-white transition-all"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleFollowUpSubmit(e as any);
                      }
                    }}
                  />
                  <button
                    type="submit"
                    disabled={!followUpQuery.trim()}
                    className="absolute inset-y-0 right-0 flex items-center pr-2 text-gray-400 hover:text-gray-600 disabled:text-gray-300 disabled:cursor-not-allowed transition-colors"
                  >
                    <PaperAirplaneIcon className="h-4 w-4" />
                  </button>
                </div>
              </form>
              
              {/* Quick follow-up suggestions */}
              <div className="mt-3">
                <div className="flex flex-wrap gap-2">
                  {quickFollowUps.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => setFollowUpQuery(suggestion)}
                      className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-50 border border-gray-200 text-gray-700 hover:bg-gray-100 hover:border-gray-300 transition-all"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}