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

export interface QueryResponse {
  query: string;
  sources: Source[];
  answer?: string;
}