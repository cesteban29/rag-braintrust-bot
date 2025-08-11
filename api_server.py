"""
FastAPI server for Braintrust RAG system
Provides REST API endpoints for the frontend to query the documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv('.env')

# Initialize FastAPI app
app = FastAPI(title="Braintrust RAG API", version="1.0.0")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients (do this once at startup)
try:
    vo_client = voyageai.Client(api_key=os.getenv("VOYAGEAI_API_KEY"))
    pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc_client.Index(os.getenv("INDEX_NAME", "braintrust-llms-txt"))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'voyage-3')
    logger.info("Successfully initialized VoyageAI and Pinecone clients")
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")
    raise

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[dict] = None

class Source(BaseModel):
    title: str
    content: str
    score: float
    section_type: str
    url: str
    summary: Optional[str] = None
    keywords: Optional[str] = None
    questions: Optional[str] = None
    date: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    sources: List[Source]
    answer: Optional[str] = None  # We could add GPT summarization later

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Braintrust RAG API"}

@app.post("/api/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """
    Search the Braintrust documentation using semantic search
    """
    try:
        # Create embedding for the query
        logger.info(f"Processing query: {request.query}")
        query_embedding = vo_client.embed(
            request.query, 
            model=EMBEDDING_MODEL, 
            input_type='query'
        ).embeddings[0]
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True,
            filter=request.filter
        )
        
        # Format sources
        sources = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            source = Source(
                title=metadata.get('title', 'Unknown'),
                content=metadata.get('content', ''),
                score=match['score'],
                section_type=metadata.get('section_type', 'general'),
                url=metadata.get('url', ''),
                summary=metadata.get('summary'),
                keywords=metadata.get('keywords'),
                questions=metadata.get('questions'),
                date=metadata.get('date')
            )
            sources.append(source)
        
        logger.info(f"Found {len(sources)} sources for query")
        
        return QueryResponse(
            query=request.query,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_index_stats():
    """
    Get statistics about the Pinecone index
    """
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
            "namespaces": stats.namespaces
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/section-types")
async def get_section_types():
    """
    Get available section types for filtering
    """
    return {
        "section_types": [
            {"value": "changelog", "label": "Changelog"},
            {"value": "api_reference", "label": "API Reference"},
            {"value": "sdk_reference", "label": "SDK Reference"},
            {"value": "tutorial", "label": "Tutorials"},
            {"value": "evaluation", "label": "Evaluations"},
            {"value": "tracing", "label": "Tracing"},
            {"value": "prompts", "label": "Prompts"},
            {"value": "general", "label": "General"},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)