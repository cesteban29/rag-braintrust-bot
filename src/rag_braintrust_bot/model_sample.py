"""
RAG Demo with Model Rotation

This script demonstrates RAG (Retrieval-Augmented Generation) capabilities while rotating through
different cost-effective models from various AI providers (all comparable to gpt-4o-mini pricing):

- OpenAI: gpt-4o-mini ($0.15/$0.60 per 1M tokens), gpt-3.5-turbo
- Anthropic: claude-3-5-haiku-20241022 (latest budget Claude)
- Together.ai: Llama 3.3, Mixtral, Qwen 2.5 (open source models)
- xAI: grok-2-1212 (current Grok model)

Each query uses the next model in rotation for comparison and testing purposes.
All models are accessed via the Braintrust AI proxy.

Usage:
    python model_sample.py                       # Run with default 5 queries
    python model_sample.py --queries 10         # Run with 10 queries
    python model_sample.py -q 3                 # Run with 3 queries
    python model_sample.py --start-model 2      # Start from model index 2
    python model_sample.py -q 8 --start-model 1 # 8 queries starting from model 1
    python model_sample.py --list-models        # List available models
"""

import os
import argparse
from braintrust import traced, wrap_openai, init_logger, current_span
from dotenv import load_dotenv
import openai
import json
import sys
import os
# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents

# Load environment variables
load_dotenv('.env')

# Validate required environment variables
required_env_vars = [
    "BRAINTRUST_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please ensure these are set in your .env.local file")
    sys.exit(1)

# Initialize Braintrust logger
logger = init_logger(project="rag-braintrust-bot")

# Initialize wrapped OpenAI client & use OpenAI API Key stored in Braintrust
client = wrap_openai(
    openai.OpenAI(
        base_url="https://braintrustproxy.com/v1",
        api_key=os.getenv('BRAINTRUST_API_KEY')
    )
)

# Define cheap models to rotate through from different providers
# All models selected for cost-effectiveness (comparable to gpt-4o-mini at $0.15/$0.60 per 1M tokens)
MODELS_ROTATION = [
    # OpenAI models (verified pricing: $0.15/$0.60 per 1M tokens)
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    
    # Anthropic models (budget option)
    "claude-3-5-haiku-20241022",  # Updated to latest Haiku
    
    # Together.ai models (open source, very cost-effective)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Updated to Llama 3.1
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",  # Added efficient Qwen model
    
    # xAI models
    "grok-2-1212",  # Updated to current Grok model
]

# Global counter for model rotation
model_counter = 0

def get_next_model():
    """Get the next model in rotation."""
    global model_counter
    model = MODELS_ROTATION[model_counter % len(MODELS_ROTATION)]
    model_counter += 1
    return model

def get_model_provider(model_name):
    """Get the provider name for a given model."""
    if model_name.startswith("gpt-"):
        return "openai"
    elif model_name.startswith("claude-"):
        return "anthropic"
    elif model_name.startswith("grok"):
        return "xai"
    elif "/" in model_name:
        # Handle Together.ai models (meta-llama/, mistralai/, Qwen/, etc.)
        return "together"
    else:
        return "unknown"

def generate_test_queries(num_queries=5):
    """
    Dynamically generate test queries using OpenAI based on available documentation.
    
    Returns:
        list: Generated test queries
    """
    print("🤖 Generating dynamic test queries...")
    try:
        # First, get a sample of available document topics
        sample_docs = get_documents("Braintrust features and capabilities")
        
        # Extract topics from the retrieved documents
        topics = []
        for doc in sample_docs["documents"][:3]:  # Use top 3 docs for context
            topics.append(f"- {doc['title']}: {doc['content'][:200]}...")
        
        topics_context = "\n".join(topics)
        
        # Generate questions using different models
        current_model = get_next_model()
        print(f"   Using model: {current_model} for query generation")
        response = client.chat.completions.create(
            model=current_model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates diverse test questions about Braintrust documentation. 
                    Create realistic questions that users might ask about Braintrust features, implementation, and best practices.
                    
                    Generate a mix of question types:
                    - How-to questions (How do I...?)
                    - What-is questions (What is...? What are...?)
                    - Explain questions (Explain how...)
                    - Best practice questions (What are the best practices for...?)
                    - Comparison questions (What's the difference between...?)
                    
                    Make sure questions are specific and actionable."""
                },
                {
                    "role": "user", 
                    "content": f"""Based on these Braintrust documentation topics:

{topics_context}

Generate {num_queries} diverse, specific questions that a developer might ask about Braintrust. 
Return only the questions, one per line, without numbering or bullet points."""
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Parse the generated questions
        generated_text = response.choices[0].message.content.strip()
        questions = [q.strip() for q in generated_text.split('\n') if q.strip()]
        
        # Filter out any empty questions and limit to requested number
        questions = [q for q in questions if len(q) > 10][:num_queries]
        
        # Fallback to default questions if generation fails
        if len(questions) < num_queries:
            fallback_questions = [
                "How do I set up logging in Braintrust?",
                "What are the main features of the Braintrust playground?", 
                "How can I evaluate a chat assistant using Braintrust?",
                "Explain how to use assertions in Braintrust",
                "What is the process for human review in Braintrust?"
            ]
            questions.extend(fallback_questions[len(questions):num_queries])
        
        return questions[:num_queries]
        
    except Exception as e:
        # Fallback to original static queries
        return [
            "How do I set up logging in Braintrust?",
            "What are the main features of the Braintrust playground?",
            "How can I evaluate a chat assistant using Braintrust?",
            "Explain how to use assertions in Braintrust",
            "What is the process for human review in Braintrust?"
        ]

# Define the RAG tool for OpenAI
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

SYSTEM_PROMPT = """You are a helpful assistant specializing in Braintrust documentation and best practices. 
Your role is to provide clear, accurate answers about Braintrust's features, tools, and implementation details.

When answering:
1. First use the get_documents tool to retrieve relevant documentation
2. Use the retrieved context to provide accurate answers
3. Be specific and include code examples when relevant
4. If you're unsure about something, say so rather than making assumptions
5. Focus on practical implementation details
"""

def handle_tool_calls(tool_calls, parent_span):
    """Handle tool calls from OpenAI."""
    retrieved_documents = []
    
    for tool_call in tool_calls:
        if tool_call.function.name == "get_documents":
            # Parse the query from the function arguments
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            
            # Create a span for document retrieval
            with logger.start_span(name="get_documents", type="tool", parent=parent_span) as span:
                # Call the RAG tool
                rag_response = get_documents(query)
                
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
                        "retrieved_documents": retrieved_documents  # Add this for scorers
                    }
                )
                
                # Log each document with its metadata
                for doc in rag_response["documents"]:
                    with logger.start_span(name=f"document_{doc['id']}", type="tool", parent=parent_span) as span:
                        span.log(
                            inputs={"query": query},
                            output={"document": doc["content"]},
                            metadata={
                                "tool_name": "get_documents",
                                "document_id": doc["id"],
                                "document_title": doc["title"],
                                "similarity_score": doc["score"],
                                "document_tags": doc["tags"]
                            }
                        )
                
                return {
                    "results": rag_response,
                    "tool_call_id": tool_call.id,
                    "retrieved_documents": retrieved_documents  # Return this for the main span
                }

@traced
def process_query(query: str):
    """Process a query using RAG and create an answer."""
    try:
        # Get the next model for this query
        current_model = get_next_model()
        print(f"   Using model: {current_model}")
        
        # First message to get tool calls
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Get initial completion with tool calls
        response = client.chat.completions.create(
            model=current_model,
            messages=messages,
            tools=rag_tool,
            tool_choice="auto"
        )

        # export the span slug
        parent_span = current_span().export()
        
        # Handle tool calls and get documents
        if response.choices[0].message.tool_calls:
            tool_response = handle_tool_calls(response.choices[0].message.tool_calls, parent_span)
        else:
            # No tool calls - create empty response
            tool_response = {"results": {"documents": []}, "tool_call_id": None, "retrieved_documents": []}
        
        # Add tool results and get final answer
        if tool_response["tool_call_id"]:
            # There were tool calls - add them to messages
            messages.extend([
                response.choices[0].message,
                {
                    "role": "tool",
                    "name": "get_documents",
                    "content": json.dumps(tool_response["results"]),
                    "tool_call_id": tool_response["tool_call_id"]
                }
            ])
            
            final_response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=1000
            )
        else:
            # No tool calls - use the original response
            final_response = response
        
        answer = final_response.choices[0].message.content
        
        # Log retrieved documents and model info to the main span for scorers to access
        current_span().log(
            metadata={
                "retrieved_documents": tool_response.get("retrieved_documents", []),
                "model": current_model,
                "provider": get_model_provider(current_model)
            }
        )
        
        # Return the answer so Braintrust can log it as the span output
        return answer
        
    except Exception as e:
        # Return empty string on error so Braintrust still has an output
        return f"Error: {str(e)}"

def main():
    """Main function to process all test queries."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Demo with Model Rotation')
    parser.add_argument(
        '--queries', '-q', 
        type=int, 
        default=5,
        help='Number of test queries to generate and process (default: 5)'
    )
    parser.add_argument(
        '--list-models', 
        action='store_true',
        help='List available models and exit'
    )
    parser.add_argument(
        '--start-model', 
        type=int, 
        default=0,
        help='Start from a specific model index (0-based, default: 0)'
    )
    
    args = parser.parse_args()
    
    # Handle list models option
    if args.list_models:
        print("Available models in rotation:")
        for i, model in enumerate(MODELS_ROTATION, 1):
            provider = get_model_provider(model)
            print(f"  {i}. {model} ({provider})")
        return
    
    # Validate arguments
    if args.queries < 1:
        print("Error: Number of queries must be at least 1")
        return
    
    if args.start_model < 0 or args.start_model >= len(MODELS_ROTATION):
        print(f"Error: Start model index must be between 0 and {len(MODELS_ROTATION)-1}")
        return
    
    # Set starting model position
    global model_counter
    model_counter = args.start_model
    
    # Generate dynamic test queries with configurable count
    test_queries = generate_test_queries(num_queries=args.queries)
    print(f"Running RAG demo with {len(test_queries)} queries...")
    print(f"Available models: {', '.join(MODELS_ROTATION)}")
    print(f"Models will cycle through {len(MODELS_ROTATION)} different providers")
    if args.start_model > 0:
        start_model_name = MODELS_ROTATION[args.start_model]
        print(f"Starting from model index {args.start_model}: {start_model_name}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query}")
        try:
            answer = process_query(query)
            print(f"  ✅ Answer generated successfully")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
        print()  # Add spacing between queries
    
    print("✅ Demo completed. Check Braintrust logs for detailed traces.")
    print(f"Total queries processed with {len(MODELS_ROTATION)} different models in rotation.")

if __name__ == "__main__":
    main() 