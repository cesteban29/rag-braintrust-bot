'use client';

import { Source } from '@/types';
import { CalendarIcon, TagIcon, QuestionMarkCircleIcon } from '@heroicons/react/24/outline';

interface SearchResultsProps {
  query: string;
  sources: Source[];
}

const getSectionTypeColor = (type: string) => {
  const colors: Record<string, string> = {
    changelog: 'bg-green-50 text-green-700 border-green-200',
    api_reference: 'bg-blue-50 text-blue-700 border-blue-200', 
    sdk_reference: 'bg-purple-50 text-purple-700 border-purple-200',
    tutorial: 'bg-orange-50 text-orange-700 border-orange-200',
    evaluation: 'bg-pink-50 text-pink-700 border-pink-200',
    tracing: 'bg-gray-50 text-gray-700 border-gray-200',
    prompts: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    release_notes: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  };
  return colors[type] || 'bg-gray-50 text-gray-700 border-gray-200';
};

const getSectionTypeLabel = (type: string) => {
  const labels: Record<string, string> = {
    changelog: 'Changelog',
    api_reference: 'API Reference',
    sdk_reference: 'SDK Reference', 
    tutorial: 'Tutorial',
    evaluation: 'Evaluation',
    tracing: 'Tracing',
    prompts: 'Prompts',
    release_notes: 'Release Notes',
  };
  return labels[type] || type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
};

function SourceCard({ source, index }: { source: Source; index: number }) {
  const sectionColor = getSectionTypeColor(source.section_type);
  const sectionLabel = getSectionTypeLabel(source.section_type);

  return (
    <div className="bg-white rounded-md border border-gray-200 overflow-hidden hover:border-gray-300 transition-colors">
      {/* Header */}
      <div className="p-4 pb-3">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-base font-semibold text-gray-900 mb-2 leading-tight">
              {source.title}
            </h3>
            <div className="flex items-center gap-2 text-sm">
              <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium border ${sectionColor}`}>
                {sectionLabel}
              </span>
              {source.date && (
                <span className="flex items-center gap-1 text-gray-500 bg-gray-50 px-2 py-1 rounded-md">
                  <CalendarIcon className="h-3 w-3" />
                  <span className="text-xs">{source.date}</span>
                </span>
              )}
            </div>
          </div>
          <div className="flex-shrink-0 ml-3">
            <div className="text-right">
              <div className="inline-flex items-center px-2 py-1 bg-gray-50 border border-gray-200 rounded-md">
                <div className="text-xs font-medium text-gray-700">
                  {(source.score * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-1">match</div>
            </div>
          </div>
        </div>

        {/* Summary */}
        {source.summary && (
          <div className="mb-4 p-3 bg-gray-50 rounded-md border border-gray-200">
            <p className="text-sm text-gray-700 leading-relaxed">
              {source.summary}
            </p>
          </div>
        )}

        {/* Content */}
        <div className="mb-4">
          <p className="text-sm text-gray-700 leading-relaxed">
            {source.content.length > 300 
              ? `${source.content.substring(0, 300)}...` 
              : source.content}
          </p>
        </div>

        {/* Keywords */}
        {source.keywords && (
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <TagIcon className="h-3 w-3 text-gray-400" />
              <span className="text-xs font-medium text-gray-600">Keywords</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {source.keywords.split(',').slice(0, 5).map((keyword, i) => (
                <span
                  key={i}
                  className="inline-flex items-center px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 rounded-md"
                >
                  {keyword.trim()}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Questions */}
        {source.questions && (
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <QuestionMarkCircleIcon className="h-3 w-3 text-gray-400" />
              <span className="text-xs font-medium text-gray-600">Related questions</span>
            </div>
            <div className="space-y-1">
              {source.questions.split('|').slice(0, 2).map((question, i) => (
                <div key={i} className="text-xs text-gray-600 bg-gray-50 p-2 rounded-md">
                  {question.trim()}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-sm font-medium text-gray-700 hover:text-black transition-colors"
          >
            <span>View source</span>
            <svg className="ml-1 w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
          <span className="text-xs text-gray-500">
            #{index + 1}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function SearchResults({ query, sources }: SearchResultsProps) {
  if (sources.length === 0) {
    return (
      <div className="mt-8 text-center py-12">
        <div className="text-gray-400 mb-4">
          <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
        <p className="text-gray-500">
          Try rephrasing your query or using different keywords.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <span className="text-sm text-gray-500">
          {sources.length} source{sources.length !== 1 ? 's' : ''} retrieved
        </span>
      </div>
      
      <div className="space-y-6">
        {sources.map((source, index) => (
          <SourceCard key={index} source={source} index={index} />
        ))}
      </div>
    </div>
  );
}