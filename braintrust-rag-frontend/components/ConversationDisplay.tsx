'use client';

import { Message } from '@/types';
import { UserIcon, SparklesIcon } from '@heroicons/react/24/outline';

interface ConversationDisplayProps {
  messages: Message[];
  loading?: boolean;
}

export default function ConversationDisplay({ messages, loading = false }: ConversationDisplayProps) {
  if (messages.length === 0 && !loading) {
    return null;
  }

  return (
    <div className="space-y-6 mb-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">Conversation</h2>
      
      <div className="space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-4 ${
              message.role === 'user' ? 'flex-row' : 'flex-row'
            }`}
          >
            {/* Avatar */}
            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
              message.role === 'user' 
                ? 'bg-blue-100 text-blue-600' 
                : 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white'
            }`}>
              {message.role === 'user' ? (
                <UserIcon className="h-5 w-5" />
              ) : (
                <SparklesIcon className="h-5 w-5" />
              )}
            </div>

            {/* Message Content */}
            <div className={`flex-1 min-w-0 ${
              message.role === 'user' ? 'max-w-3xl' : 'max-w-4xl'
            }`}>
              <div className={`rounded-2xl px-6 py-4 ${
                message.role === 'user'
                  ? 'bg-blue-50 border border-blue-200'
                  : 'bg-white border border-gray-200 shadow-sm'
              }`}>
                <div className={`text-sm font-semibold mb-2 ${
                  message.role === 'user' ? 'text-blue-700' : 'text-gray-700'
                }`}>
                  {message.role === 'user' ? 'You' : 'Braintrust Assistant'}
                </div>
                <div className={`prose prose-sm max-w-none ${
                  message.role === 'user' 
                    ? 'text-blue-900' 
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
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 text-white flex items-center justify-center">
              <SparklesIcon className="h-5 w-5" />
            </div>
            <div className="flex-1 min-w-0 max-w-4xl">
              <div className="rounded-2xl px-6 py-4 bg-white border border-gray-200 shadow-sm">
                <div className="text-sm font-semibold mb-2 text-gray-700">
                  Braintrust Assistant
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-sm text-gray-500">Generating response...</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}