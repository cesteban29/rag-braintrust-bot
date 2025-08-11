"""
FastAPI server for Braintrust RAG system
Provides REST API endpoints for the frontend to query the documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone
import logging
import openai
import json

# Load environment variables
load_dotenv('.env')

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents_handler

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
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'voyage-3')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    logger.info("Successfully initialized VoyageAI, Pinecone, and OpenAI clients")
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")
    raise

# Request/Response models
class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filter: Optional[dict] = None
    conversation_history: Optional[List[Message]] = None

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
    answer: str
    conversation_history: List[Message]

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant specializing in Braintrust documentation and best practices. 
Your role is to provide clear, accurate answers about Braintrust's features, tools, and implementation details.

When answering:
1. Use the retrieved documentation context to provide accurate answers
2. Be specific and include code examples when relevant from the documentation
3. If you're unsure about something, say so rather than making assumptions
4. Focus on practical implementation details
5. Structure your responses clearly with proper markdown formatting
6. If the user asks about something not covered in the retrieved documents, clearly state that

FORMATTING GUIDELINES:
- Use proper markdown headers (# ## ###) instead of single # on empty lines
- Use code blocks (```language\ncode\n```) for multi-line code examples
- Use `backticks` for inline code, function names, and technical terms
- Use **bold** for important concepts and *italics* for emphasis
- Use bullet points (-) and numbered lists (1.) for step-by-step instructions
- Include clickable links in [text](url) format when referencing external resources

You have access to retrieved documents that contain relevant information to answer the user's question."""

def generate_rag_response(query: str, conversation_history: List[Message] = None) -> tuple[str, List[Source]]:
    """Generate a RAG response using the existing retrieval tool and LLM."""
    try:
        # Use the existing retrieval handler
        rag_response = get_documents_handler(query)
        
        # Convert to our Source format
        sources = []
        for doc in rag_response["documents"]:
            source = Source(
                title=doc["title"],
                content=doc["content"],
                score=doc["score"],
                section_type=doc["section_type"],
                url=doc["url"],
                summary=doc.get("summary"),
                keywords=doc.get("keywords"), 
                questions=doc.get("questions"),
                date=doc.get("date")
            )
            sources.append(source)
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(rag_response["documents"], 1):
            context_part = f"## Document {i}: {doc['title']}\n"
            context_part += f"**Section:** {doc['section_type']}\n"
            context_part += f"**Content:** {doc['content']}\n"
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        # Build messages for the conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current query with context
        user_message = f"User Query: {query}\n\n## Retrieved Documentation Context:\n{context}\n\nPlease provide a comprehensive answer based on the retrieved documentation context above."
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1500,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content or "I apologize, but I couldn't generate a response."
        return answer, sources
        
    except Exception as e:
        logger.error(f"RAG response generation error: {e}")
        error_msg = f"I apologize, but I encountered an error while generating a response: {str(e)}"
        return error_msg, []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Braintrust RAG API"}

@app.post("/api/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """
    Search the Braintrust documentation using semantic search and generate RAG response
    """
    try:
        logger.info(f"Processing RAG query: {request.query}")
        
        # Generate RAG response using existing tools
        answer, sources = generate_rag_response(request.query, request.conversation_history)
        
        # Build updated conversation history
        updated_history = []
        if request.conversation_history:
            updated_history.extend(request.conversation_history)
        
        # Add the current exchange
        updated_history.extend([
            Message(role="user", content=request.query),
            Message(role="assistant", content=answer)
        ])
        
        logger.info(f"Generated RAG response with {len(sources)} sources")
        
        return QueryResponse(
            query=request.query,
            sources=sources,
            answer=answer,
            conversation_history=updated_history
        )
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
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