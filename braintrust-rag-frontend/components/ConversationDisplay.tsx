'use client';

import { Message } from '@/types';
import { UserIcon, SparklesIcon, PaperAirplaneIcon, HandThumbUpIcon, HandThumbDownIcon } from '@heroicons/react/24/outline';
import { HandThumbUpIcon as HandThumbUpIconSolid, HandThumbDownIcon as HandThumbDownIconSolid } from '@heroicons/react/24/solid';
import React, { useState, FormEvent } from 'react';

interface ConversationDisplayProps {
  messages: Message[];
  loading?: boolean;
  onFollowUp?: (query: string) => void;
  conversationId?: string;
}

export default function ConversationDisplay({ messages, loading = false, onFollowUp, conversationId }: ConversationDisplayProps) {
  const [followUpQuery, setFollowUpQuery] = useState('');
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);

  // Reset feedback when conversationId changes
  React.useEffect(() => {
    setFeedback(null);
  }, [conversationId]);

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

  const handleFeedback = async (feedbackType: 'positive' | 'negative') => {
    if (!conversationId || feedbackSubmitting) return;
    
    setFeedbackSubmitting(true);
    try {
      const response = await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          feedback: feedbackType
        }),
      });

      if (response.ok) {
        setFeedback(feedbackType);
      } else {
        const errorText = await response.text();
        console.error('Failed to submit feedback:', response.status, errorText);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setFeedbackSubmitting(false);
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
                        __html: (() => {
                          let html = message.content;
                          
                          // First, protect code blocks by replacing them with placeholders
                          const codeBlocks: string[] = [];
                          html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                            const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
                            codeBlocks.push(`<pre class="bg-gray-900 text-gray-100 rounded-lg p-4 my-4 overflow-x-auto"><code class="font-mono text-sm">${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`);
                            return placeholder;
                          });
                          
                          // Handle single backticks
                          html = html.replace(/`([^`\n]+)`/g, '<code class="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-gray-800">$1</code>');
                          
                          // Handle headers (process from most specific to least specific)
                          html = html.replace(/^###### (.*$)/gm, '<h6 class="text-xs font-medium text-gray-700 mt-2 mb-1">$1</h6>');
                          html = html.replace(/^##### (.*$)/gm, '<h5 class="text-sm font-medium text-gray-800 mt-3 mb-1">$1</h5>');
                          html = html.replace(/^#### (.*$)/gm, '<h4 class="text-base font-semibold text-gray-900 mt-4 mb-2">$1</h4>');
                          html = html.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-900 mt-5 mb-3">$1</h3>');
                          html = html.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mt-6 mb-3">$1</h2>');
                          html = html.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-gray-900 mt-6 mb-4">$1</h1>');
                          
                          // Handle bold and italic
                          html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
                          html = html.replace(/\*(.*?)\*/g, '<em class="italic">$1</em>');
                          
                          // Handle links
                          html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-black hover:text-gray-600 underline">$1</a>');
                          
                          // Handle bullet points
                          html = html.replace(/^- (.*$)/gm, '<li class="ml-4">$1</li>');
                          html = html.replace(/(<li.*<\/li>)/s, '<ul class="list-disc list-inside mb-3 space-y-1">$1</ul>');
                          
                          // Handle numbered lists
                          html = html.replace(/^\d+\. (.*$)/gm, '<li class="ml-4">$1</li>');
                          html = html.replace(/(<li.*<\/li>)/s, '<ol class="list-decimal list-inside mb-3 space-y-1">$1</ol>');
                          
                          // Handle line breaks and paragraphs (but not for placeholders)
                          html = html.replace(/\n\n/g, '</p><p class="mb-3">');
                          html = html.replace(/^(?!<[huo]|__CODE_BLOCK_|<li)(.+$)/gm, '<p class="mb-3">$1</p>');
                          
                          // Clean up empty paragraphs
                          html = html.replace(/<p[^>]*><\/p>/g, '');
                          
                          // Restore code blocks
                          codeBlocks.forEach((codeBlock, index) => {
                            html = html.replace(`__CODE_BLOCK_${index}__`, codeBlock);
                          });
                          
                          return html;
                        })()
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

      {/* Feedback Section - Only show after conversation and when not loading */}
      {messages.length > 0 && !loading && conversationId && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Was this conversation helpful?
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleFeedback('positive')}
                disabled={feedbackSubmitting}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  feedback === 'positive'
                    ? 'bg-green-50 text-green-700 border border-green-200'
                    : 'bg-gray-50 text-gray-700 border border-gray-200 hover:bg-gray-100 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {feedback === 'positive' ? (
                  <HandThumbUpIconSolid className="h-4 w-4" />
                ) : (
                  <HandThumbUpIcon className="h-4 w-4" />
                )}
                <span>Yes</span>
              </button>
              <button
                onClick={() => handleFeedback('negative')}
                disabled={feedbackSubmitting}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                  feedback === 'negative'
                    ? 'bg-red-50 text-red-700 border border-red-200'
                    : 'bg-gray-50 text-gray-700 border border-gray-200 hover:bg-gray-100 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {feedback === 'negative' ? (
                  <HandThumbDownIconSolid className="h-4 w-4" />
                ) : (
                  <HandThumbDownIcon className="h-4 w-4" />
                )}
                <span>No</span>
              </button>
              {feedback && (
                <span className="text-xs text-gray-500 ml-2">
                  Thank you for your feedback!
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}