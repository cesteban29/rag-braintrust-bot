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
from datetime import datetime
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
conversation_spans = {}  # Store the actual span objects to keep them open

# System prompt for the assistant
SYSTEM_PROMPT = """You are a helpful assistant specializing in Braintrust documentation and best practices. 
Your role is to provide clear, accurate answers about Braintrust's features, tools, and implementation details.

When answering:
1. First use the get_documents tool to retrieve relevant documentation
2. Use the retrieved context to provide accurate answers
3. Be specific and include code examples when relevant
4. If you're unsure about something, say so rather than making assumptions
5. Focus on practical implementation details
"""

# RAG tool definition
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

def get_next_model():
    """Get the next model for rotation (simplified for API server)."""
    return LLM_MODEL

def get_model_provider(model_name: str) -> str:
    """Get the provider name for a given model."""
    if model_name.startswith("gpt-"):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic" 
    elif model_name.startswith("grok"):
        return "xai"
    elif "/" in model_name:
        return "together"
    else:
        return "unknown"

async def process_conversation_turn(query: str, conversation_history: list, parent_span=None) -> str:
    """Process a single conversation turn with tool calls, mimicking rag_simulation.py behavior.
    
    Args:
        query: The user's query
        conversation_history: Previous conversation messages
        parent_span: Optional parent Braintrust span object for creating child spans
    """
    try:
        # Get the model for this query
        current_model = get_next_model()
        
        # Build messages for the conversation
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Get initial completion with tool calls
        response = openai_client.chat.completions.create(
            model=current_model,
            messages=messages,
            tools=rag_tool,
            tool_choice="auto"
        )
        
        # Handle tool calls and get documents
        retrieved_documents = []
        if response.choices[0].message.tool_calls:
            tool_response = await handle_tool_calls_async(response.choices[0].message.tool_calls, parent_span)
            retrieved_documents = tool_response.get("retrieved_documents", [])
            
            # Add tool calls to messages
            messages.extend([
                response.choices[0].message,
                {
                    "role": "tool",
                    "name": "get_documents",
                    "content": json.dumps(tool_response["results"]),
                    "tool_call_id": tool_response["tool_call_id"]
                }
            ])
            
            final_response = openai_client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=1000
            )
            answer = final_response.choices[0].message.content
        else:
            # No tool calls - use the original response
            answer = response.choices[0].message.content
        
        return answer or ""
        
    except Exception as e:
        return f"Error: {str(e)}"

async def handle_tool_calls_async(tool_calls, parent_span=None) -> dict:
    """Handle tool calls from OpenAI and return retrieved documents (async version).
    
    Args:
        tool_calls: Tool calls from OpenAI response
        parent_span: Optional parent Braintrust span object for creating child spans
    """
    import json
    
    retrieved_documents = []
    
    for tool_call in tool_calls:
        if tool_call.function.name == "get_documents":
            # Parse the query from the function arguments
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            
            # Create a span for document retrieval (only use parent if provided)
            span_kwargs = {"name": "get_documents", "type": "tool"}
            if parent_span:
                span_kwargs["parent"] = parent_span
            
            with braintrust_logger.start_span(**span_kwargs) as span:
                # Call the RAG tool
                rag_response = get_documents_handler(query)
                
                # Prepare retrieved documents for scorers
                for doc in rag_response["documents"]:
                    retrieved_documents.append({
                        "id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "score": doc["score"],
                        "tags": doc.get("tags", [])
                    })
                
                # Log the input query & document ids, plus retrieved_documents for scorers
                span.log(
                    inputs={"query": query},
                    metadata={
                        "tool_name": "get_documents", 
                        "document_ids": [doc['id'] for doc in rag_response["documents"]],
                        "retrieved_documents": retrieved_documents
                    }
                )
                
                return {
                    "results": rag_response,
                    "tool_call_id": tool_call.id,
                    "retrieved_documents": retrieved_documents
                }

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

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: str  # 'positive' or 'negative'
    comment: Optional[str] = None

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
        import uuid
        import json
        
        # Determine if this is a new conversation or continuation
        is_new_conversation = not bool(request.conversation_history)
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        if is_new_conversation:
            # Start a new conversation trace (without 'with' statement to keep it open)
            conversation_span = braintrust_logger.start_span(
                name="rag_conversation", 
                type="conversation",
                inputs={"query": request.query}
            )
            
            # Store both the span object and its export
            conversation_spans[conversation_id] = conversation_span
            conversation_root_spans[conversation_id] = conversation_span.export()
            
            # Process the query within the conversation span context
            answer = await process_conversation_turn(
                request.query, 
                [], 
                conversation_span  # Pass the span object
            )
            
            # Update the root span with the final output
            conversation_span.log(
                metadata={"answer": answer}
            )
        else:
            # Continue existing conversation
            if conversation_id not in conversation_spans:
                raise HTTPException(status_code=400, detail="Conversation not found")
            
            # Get the conversation span to use as parent
            conversation_span = conversation_spans[conversation_id]
            
            # Build conversation history for context
            conversation_history = []
            for msg in request.conversation_history:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Process the follow-up query as a child span
            with braintrust_logger.start_span(
                name="rag_followup",
                type="followup", 
                parent=conversation_span,
                inputs={"query": request.query, "conversation_history": conversation_history}
            ) as followup_span:
                # Pass the followup_span for child spans
                answer = await process_conversation_turn(
                    request.query,
                    conversation_history, 
                    followup_span
                )
                
                followup_span.log(metadata={"answer": answer})
            
            # Update root span with latest conversation state
            updated_conversation = conversation_history + [
                {"role": "user", "content": request.query},
                {"role": "assistant", "content": answer}
            ]
            
            # Update the conversation span with latest info
            conversation_span.log(
                metadata={
                    "latest_answer": answer,
                    "conversation_turn": len(conversation_history) // 2 + 1
                }
            )
        
        # Get sources for the latest query (outside the trace spans)
        rag_response = get_documents_handler(request.query)
        sources = []
        for doc in rag_response["documents"]:
            source = Source(
                title=doc["title"],
                content=doc["content"],
                score=doc["score"],
                section_type=doc.get("section_type", "general"),
                url=doc.get("url", ""),
                summary=doc.get("summary"),
                keywords=doc.get("keywords"),
                questions=doc.get("questions"),
                date=doc.get("date")
            )
            sources.append(source)
        
        # Build updated conversation history
        if is_new_conversation:
            updated_history = [
                Message(role="user", content=request.query),
                Message(role="assistant", content=answer)
            ]
        else:
            updated_history = list(request.conversation_history)
            updated_history.extend([
                Message(role="user", content=request.query),
                Message(role="assistant", content=answer)
            ])
        
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

@app.post("/api/feedback")
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit feedback for a conversation"""
    try:
        if feedback_request.conversation_id not in conversation_root_spans:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        parent_span_export = conversation_root_spans[feedback_request.conversation_id]
        
        # Convert feedback to score (1 for positive, 0 for negative)
        score = 1.0 if feedback_request.feedback == 'positive' else 0.0
        
        # The parent_span_export is a string ID from span.export()
        # We need to use this ID directly for log_feedback
        
        # Log feedback with Braintrust using the logger's log_feedback method
        # Use keyword arguments as shown in the documentation
        comment_text = feedback_request.comment if feedback_request.comment else f"User feedback: {feedback_request.feedback}"
        
        logger.info(f"Parent span export type: {type(parent_span_export)}, value: {parent_span_export}")
        
        # Log feedback - this can be called multiple times if user changes their mind
        # Braintrust will update the feedback score if called again with the same span ID
        braintrust_logger.log_feedback(
            id=parent_span_export,
            scores={
                "user_rating": score
            },
            comment=comment_text,
            metadata={
                "feedback_type": feedback_request.feedback,
                "conversation_id": feedback_request.conversation_id,
                "timestamp": str(datetime.now()),
                "feedback_updated": feedback_request.conversation_id in conversation_traces and conversation_traces.get(feedback_request.conversation_id) is not None
            }
        )
        
        # Track that feedback was given but don't close the span
        conversation_traces[feedback_request.conversation_id] = feedback_request.feedback
        
        logger.info(f"Feedback submitted for conversation {feedback_request.conversation_id}: {feedback_request.feedback}")
        
        return {
            "status": "feedback_submitted",
            "conversation_id": feedback_request.conversation_id,
            "feedback": feedback_request.feedback
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
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
            del conversation_root_spans[conversation_id]
            cleared = True
        if conversation_id in conversation_spans:
            # End the span properly before removing
            try:
                conversation_spans[conversation_id].end()
            except:
                pass  # Span might already be ended
            del conversation_spans[conversation_id]
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