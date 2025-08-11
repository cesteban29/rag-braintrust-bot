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
from braintrust import traced, wrap_openai, init_logger, current_span

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

# Initialize Braintrust logger
braintrust_logger = init_logger(project="rag-braintrust-bot-frontend")

# Store conversation trace exports and root spans (in production, use Redis or similar)
conversation_traces = {}
conversation_root_spans = {}

# Validate required environment variables for Braintrust
required_env_vars = ["BRAINTRUST_API_KEY", "VOYAGEAI_API_KEY", "PINECONE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients (do this once at startup)
try:
    vo_client = voyageai.Client(api_key=os.getenv("VOYAGEAI_API_KEY"))
    pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc_client.Index(os.getenv("INDEX_NAME", "braintrust-llms-txt"))
    # Use wrapped OpenAI client for Braintrust tracking
    openai_client = wrap_openai(
        openai.OpenAI(
            base_url="https://braintrustproxy.com/v1",
            api_key=os.getenv('BRAINTRUST_API_KEY')
        )
    )
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
    conversation_id: Optional[str] = None  # To track conversation traces

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
    conversation_id: str  # Return the conversation ID for tracking

# System prompt for the RAG assistant
SYSTEM_PROMPT = """You are a helpful assistant specializing in Braintrust documentation and best practices. 
Your role is to provide clear, accurate answers about Braintrust's features, tools, and implementation details.

When answering:
1. First use the get_documents tool to retrieve relevant documentation
2. Use the retrieved context to provide accurate answers
3. Be specific and include code examples when relevant from the documentation
4. If you're unsure about something, say so rather than making assumptions
5. Focus on practical implementation details
6. Structure your responses clearly with proper markdown formatting

FORMATTING GUIDELINES:
- Use proper markdown headers (# ## ###) instead of single # on empty lines
- Use code blocks (```language\ncode\n```) for multi-line code examples
- Use `backticks` for inline code, function names, and technical terms
- Use **bold** for important concepts and *italics* for emphasis
- Use bullet points (-) and numbered lists (1.) for step-by-step instructions
- Include clickable links in [text](url) format when referencing external resources"""

# RAG tool definition for OpenAI function calling
rag_tool = [
    {
        "type": "function",
        "function": {
            "name": "get_documents",
            "description": "Retrieve relevant documents from the Braintrust documentation based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def handle_tool_calls(tool_calls, parent_export) -> Dict[str, Any]:
    """Handle tool calls from OpenAI and return retrieved documents."""
    retrieved_documents = []
    tool_responses = []
    
    for tool_call in tool_calls:
        if tool_call.function.name == "get_documents":
            # Parse the query from the function arguments
            import json
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            
            # Create a traced function for this tool call
            @traced(name="get_documents", type="tool", parent=parent_export)
            def call_get_documents():
                # Call the RAG tool
                rag_response = get_documents_handler(query)
                
                # Prepare retrieved documents for scorers
                docs = []
                for doc in rag_response["documents"]:
                    docs.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "score": doc["score"],
                        "section_type": doc["section_type"],
                        "url": doc.get("url", ""),
                        "tags": doc.get("tags", [])
                    })
                
                # Log the input query & document ids
                current_span().log(
                    inputs={"query": query},
                    metadata={
                        "tool_name": "get_documents", 
                        "document_ids": [doc['id'] for doc in rag_response["documents"]],
                        "retrieved_documents": docs,
                        "num_documents": len(rag_response["documents"])
                    }
                )
                
                return rag_response, docs
            
            # Call the traced function
            rag_response, docs = call_get_documents()
            retrieved_documents.extend(docs)
            
            # Format response for OpenAI
            tool_responses.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "get_documents",
                "content": json.dumps(rag_response)
            })
    
    return {
        "tool_responses": tool_responses,
        "retrieved_documents": retrieved_documents
    }

@traced(name="llm_generation", type="llm")
def generate_rag_response(query: str, conversation_history: List[Message] = None) -> tuple[str, List[Source]]:
    """Generate a RAG response using OpenAI function calling."""
    try:
        # Build messages for the conversation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user query
        messages.append({"role": "user", "content": query})
        
        # Get initial completion with tool calls
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=rag_tool,
            tool_choice="auto",
            max_tokens=1500,
            temperature=0.1
        )
        
        # Handle tool calls and get documents
        retrieved_documents = []
        sources = []
        
        if response.choices[0].message.tool_calls:
            # Pass current span export as parent for tool calls
            tool_result = handle_tool_calls(response.choices[0].message.tool_calls, current_span().export())
            retrieved_documents = tool_result.get("retrieved_documents", [])
            
            # Convert retrieved documents to Source format
            for doc in retrieved_documents:
                source = Source(
                    title=doc["title"],
                    content=doc["content"],
                    score=doc["score"],
                    section_type=doc["section_type"],
                    url=doc.get("url", ""),
                    summary=doc.get("summary"),
                    keywords=doc.get("keywords"), 
                    questions=doc.get("questions"),
                    date=doc.get("date")
                )
                sources.append(source)
            
            # Add tool calls and responses to message history
            messages.append(response.choices[0].message)
            messages.extend(tool_result["tool_responses"])
            
            # Get final response after tool calls
            final_response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            answer = final_response.choices[0].message.content
        else:
            # No tool calls - use the original response
            answer = response.choices[0].message.content
        
        answer = answer or "I apologize, but I couldn't generate a response."
        
        # Log metadata for the LLM generation
        current_span().log(
            metadata={
                "model": LLM_MODEL,
                "conversation_type": "multi_turn" if conversation_history else "single_turn",
                "num_turns": len(conversation_history) // 2 + 1 if conversation_history else 1,
                "response_length": len(answer),
                "num_sources": len(sources),
                "used_tools": bool(response.choices[0].message.tool_calls)
            }
        )
        
        return answer, sources
        
    except Exception as e:
        logger.error(f"RAG response generation error: {e}")
        error_msg = f"I apologize, but I encountered an error while generating a response: {str(e)}"
        
        # Log error as child span if parent exists
        if parent_span:
            with braintrust_logger.start_span(
                name="rag_error", 
                type="error", 
                parent=parent_span
            ) as error_span:
                error_span.log(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "model": LLM_MODEL,
                        "conversation_type": "multi_turn" if conversation_history else "single_turn"
                    }
                )
        
        return error_msg, []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Braintrust RAG API"}

@traced(name="rag_followup", type="followup")
def process_followup(request: QueryRequest, conversation_id: str):
    """Process a follow-up message as a child span of the conversation trace"""
    
    # Generate RAG response (will be child span due to @traced)
    answer, sources = generate_rag_response(
        request.query, 
        request.conversation_history
    )
    
    # Build updated conversation history
    updated_history = []
    if request.conversation_history:
        updated_history.extend(request.conversation_history)
    
    updated_history.extend([
        Message(role="user", content=request.query),
        Message(role="assistant", content=answer)
    ])
    
    # Log follow-up metadata
    current_span().log(
        inputs={"followup_query": request.query},
        metadata={
            "conversation_id": conversation_id,
            "turn_number": len(request.conversation_history) // 2 + 1,
            "query_length": len(request.query),
            "response_length": len(answer),
            "sources_retrieved": len(sources),
            "updated_conversation_length": len(updated_history)
        }
    )
    
    return answer, sources, updated_history

@app.post("/api/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """
    Search the Braintrust documentation using semantic search and generate RAG response
    """
    try:
        logger.info(f"Processing RAG query: {request.query}")
        import uuid
        
        # Determine if this is a new conversation or continuation
        is_new_conversation = not bool(request.conversation_history)
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        if is_new_conversation:
            # Create a traced function for the new conversation
            @traced(name="rag_conversation", type="conversation")
            def start_conversation():
                # Log the initial user input
                current_span().log(
                    inputs={
                        "initial_query": request.query,
                        "top_k": request.top_k
                    },
                    metadata={
                        "conversation_id": conversation_id,
                        "conversation_type": "new",
                        "query_length": len(request.query),
                        "timestamp": __import__('datetime').datetime.now().isoformat()
                    }
                )
                
                # Store the conversation trace
                parent_export = current_span().export()
                conversation_traces[conversation_id] = parent_export
                conversation_root_spans[conversation_id] = current_span()
                
                # Generate RAG response (will be child span)
                answer, sources = generate_rag_response(
                    request.query, 
                    request.conversation_history
                )
                
                # Build conversation history
                updated_history = [
                    Message(role="user", content=request.query),
                    Message(role="assistant", content=answer)
                ]
                
                # Update the root span with the first output
                current_span().log(
                    metadata={
                        "final_answer": answer,
                        "conversation_history": [msg.model_dump() for msg in updated_history],
                        "num_sources": len(sources),
                        "total_turns": len(updated_history),
                        "response_length": len(answer),
                        "sources_retrieved": len(sources)
                    }
                )
                
                return answer, sources, updated_history
            
            # Start the conversation
            answer, sources, updated_history = start_conversation()
        else:
            # This is a follow-up - use the stored conversation trace as parent
            if conversation_id not in conversation_traces:
                raise HTTPException(status_code=400, detail="Invalid conversation ID")
            
            parent_export = conversation_traces[conversation_id]
            
            # Create traced function with parent
            process_followup_with_parent = traced(parent=parent_export)(process_followup)
            
            # Process follow-up as child span
            answer, sources, updated_history = process_followup_with_parent(request, conversation_id)
            
            # Update the root span with the latest output
            if conversation_id in conversation_root_spans:
                root_span = conversation_root_spans[conversation_id]
                root_span.log(
                    metadata={
                        "final_answer": answer,
                        "conversation_history": [msg.model_dump() for msg in updated_history],
                        "num_sources": len(sources),
                        "total_turns": len(updated_history),
                        "response_length": len(answer),
                        "sources_retrieved": len(sources),
                        "last_updated": __import__('datetime').datetime.now().isoformat()
                    }
                )
        
        logger.info(f"Generated RAG response with {len(sources)} sources")
        
        return QueryResponse(
            query=request.query,
            sources=sources,
            answer=answer,
            conversation_history=updated_history,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        
        # Log the error in a separate span
        with braintrust_logger.start_span(name="api_error", type="error") as error_span:
            error_span.log(
                inputs={"query": request.query},
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_success": False,
                    "is_follow_up": bool(request.conversation_history)
                }
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-conversation")
async def clear_conversation(conversation_id: str = None):
    """Clear a specific conversation from the trace store"""
    if conversation_id:
        cleared = False
        if conversation_id in conversation_traces:
            del conversation_traces[conversation_id]
            cleared = True
        if conversation_id in conversation_root_spans:
            # Make sure the span is ended before removing
            root_span = conversation_root_spans[conversation_id]
            try:
                root_span.end()
            except:
                pass  # Span might already be ended
            del conversation_root_spans[conversation_id]
            cleared = True
        if cleared:
            return {"status": "conversation cleared", "conversation_id": conversation_id}
    return {"status": "conversation not found or already cleared"}

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