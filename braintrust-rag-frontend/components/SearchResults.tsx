'use client';

import { Source } from '@/types';
import { CalendarIcon, TagIcon, QuestionMarkCircleIcon } from '@heroicons/react/24/outline';

interface SearchResultsProps {
  query: string;
  sources: Source[];
}

const getSectionTypeColor = (type: string) => {
  const colors: Record<string, string> = {
    changelog: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    api_reference: 'bg-blue-100 text-blue-700 border-blue-200', 
    sdk_reference: 'bg-purple-100 text-purple-700 border-purple-200',
    tutorial: 'bg-orange-100 text-orange-700 border-orange-200',
    evaluation: 'bg-pink-100 text-pink-700 border-pink-200',
    tracing: 'bg-indigo-100 text-indigo-700 border-indigo-200',
    prompts: 'bg-amber-100 text-amber-700 border-amber-200',
    release_notes: 'bg-cyan-100 text-cyan-700 border-cyan-200',
  };
  return colors[type] || 'bg-slate-100 text-slate-700 border-slate-200';
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
    <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100 overflow-hidden group">
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 mb-3 leading-tight group-hover:text-indigo-600 transition-colors">
              {source.title}
            </h3>
            <div className="flex items-center gap-3 text-sm">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${sectionColor}`}>
                {sectionLabel}
              </span>
              {source.date && (
                <span className="flex items-center gap-1.5 text-gray-500 bg-gray-50 px-2 py-1 rounded-full">
                  <CalendarIcon className="h-3.5 w-3.5" />
                  <span className="text-xs font-medium">{source.date}</span>
                </span>
              )}
            </div>
          </div>
          <div className="flex-shrink-0 ml-4">
            <div className="text-right">
              <div className="inline-flex items-center px-3 py-1.5 bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-full">
                <div className="text-xs font-semibold text-indigo-700">
                  {(source.score * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-1">relevance</div>
            </div>
          </div>
        </div>

        {/* Summary */}
        {source.summary && (
          <div className="mb-5 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border-l-4 border-indigo-400">
            <p className="text-sm text-gray-700 leading-relaxed font-medium">
              💡 {source.summary}
            </p>
          </div>
        )}

        {/* Content */}
        <div className="mb-5">
          <p className="text-gray-700 leading-relaxed">
            {source.content.length > 350 
              ? `${source.content.substring(0, 350)}...` 
              : source.content}
          </p>
        </div>

        {/* Keywords */}
        {source.keywords && (
          <div className="mb-5">
            <div className="flex items-center gap-2 mb-3">
              <TagIcon className="h-4 w-4 text-indigo-400" />
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Keywords</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {source.keywords.split(',').slice(0, 6).map((keyword, i) => (
                <span
                  key={i}
                  className="inline-flex items-center px-2.5 py-1 text-xs font-medium bg-gray-50 text-gray-700 border border-gray-200 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  {keyword.trim()}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Questions */}
        {source.questions && (
          <div className="mb-5">
            <div className="flex items-center gap-2 mb-3">
              <QuestionMarkCircleIcon className="h-4 w-4 text-indigo-400" />
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wider">Answers these questions</span>
            </div>
            <div className="space-y-2">
              {source.questions.split('|').slice(0, 3).map((question, i) => (
                <div key={i} className="flex items-start gap-2 text-sm text-gray-600 bg-gray-50 p-2 rounded-md">
                  <span className="text-indigo-400 font-bold text-xs mt-0.5">Q:</span>
                  <span className="font-medium">{question.trim()}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-4 bg-gray-50 border-t border-gray-100">
        <div className="flex items-center justify-between">
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors group"
          >
            <span>View original source</span>
            <svg className="ml-1 w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
          <span className="text-xs text-gray-400">
            Result #{index + 1}
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
    <div className="mt-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">
          Results for "{query}"
        </h2>
        <span className="text-sm text-gray-500">
          {sources.length} result{sources.length !== 1 ? 's' : ''} found
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