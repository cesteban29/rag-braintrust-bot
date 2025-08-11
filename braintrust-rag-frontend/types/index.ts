export interface Source {
  title: string;
  content: string;
  score: number;
  section_type: string;
  url: string;
  summary?: string;
  keywords?: string;
  questions?: string;
  date?: string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueryResponse {
  query: string;
  sources: Source[];
  answer: string;
  conversation_history: Message[];
}