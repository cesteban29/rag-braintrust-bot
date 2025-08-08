"""
RAG Simulation with Model Rotation and Multi-turn Support

This consolidated script combines the functionality of rag_demo.py and model_sample.py
to provide comprehensive RAG simulations with online scoring support for Braintrust.

Features:
- Model rotation through cost-effective models from various providers
- Multi-turn conversation simulation
- Single-turn and multi-turn RAG conversations  
- Proper metadata tracking for online Braintrust scorers
- Configurable number of logs per model (default: 5)
- Support for GitHub Actions scheduling

Usage:
    python rag_simulation.py                          # 5 logs per model (default)
    python rag_simulation.py --logs-per-model 10     # 10 logs per model
    python rag_simulation.py --mode single           # Single-turn only
    python rag_simulation.py --mode multi            # Multi-turn only  
    python rag_simulation.py --mode both             # Both single and multi-turn (default)
    python rag_simulation.py --list-models           # List available models
"""

import os
import argparse
import json
import random
import sys
from typing import List, Dict, Any, Optional
from braintrust import traced, wrap_openai, init_logger, current_span
from dotenv import load_dotenv
import openai

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents

# Load environment variables - look for .env in project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Validate required environment variables
required_env_vars = [
    "BRAINTRUST_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please ensure these are set in your .env file")
    sys.exit(1)

# Initialize Braintrust logger
logger = init_logger(project="rag-braintrust-bot")

# Initialize wrapped OpenAI client for Braintrust proxy
client = wrap_openai(
    openai.OpenAI(
        base_url="https://braintrustproxy.com/v1",
        api_key=os.getenv('BRAINTRUST_API_KEY')
    )
)

# Define models for rotation (cost-effective options)
MODELS_ROTATION = [
    # OpenAI models
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    
    # Anthropic models
    "claude-3-5-haiku-20241022",
    
    # Together.ai models (open source)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1", 
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    
    # xAI models
    "grok-2-1212",
]

# Global counter for model rotation
model_counter = 0

def get_next_model():
    """Get the next model in rotation."""
    global model_counter
    model = MODELS_ROTATION[model_counter % len(MODELS_ROTATION)]
    model_counter += 1
    return model

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

SYSTEM_PROMPT = """You are a helpful assistant specializing in Braintrust documentation and best practices. 
Your role is to provide clear, accurate answers about Braintrust's features, tools, and implementation details.

When answering:
1. First use the get_documents tool to retrieve relevant documentation
2. Use the retrieved context to provide accurate answers
3. Be specific and include code examples when relevant
4. If you're unsure about something, say so rather than making assumptions
5. Focus on practical implementation details
"""

def generate_test_queries(num_queries: int = 5) -> List[str]:
    """
    Dynamically generate test queries using OpenAI based on available documentation.
    
    Returns:
        list: Generated test queries
    """
    print(f"🤖 Generating {num_queries} dynamic test queries...")
    try:
        # First, get a sample of available document topics
        sample_docs = get_documents("Braintrust features and capabilities")
        
        # Extract topics from the retrieved documents
        topics = []
        for doc in sample_docs["documents"][:3]:  # Use top 3 docs for context
            topics.append(f"- {doc['title']}: {doc['content'][:200]}...")
        
        topics_context = "\n".join(topics)
        
        # Generate questions using a model
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
                "What is the process for human review in Braintrust?",
                "How do I create custom scorers in Braintrust?",
                "What are the best practices for dataset management?",
                "How does Braintrust handle model comparison?",
            ]
            questions.extend(fallback_questions[len(questions):num_queries])
        
        return questions[:num_queries]
        
    except Exception as e:
        print(f"   Warning: Query generation failed ({e}), using fallback questions")
        # Fallback to default questions
        fallback_questions = [
            "How do I set up logging in Braintrust?",
            "What are the main features of the Braintrust playground?",
            "How can I evaluate a chat assistant using Braintrust?",
            "Explain how to use assertions in Braintrust",
            "What is the process for human review in Braintrust?",
        ]
        return fallback_questions[:num_queries]

def create_multi_turn_scenarios(num_scenarios: int = 2) -> List[List[str]]:
    """
    Create multi-turn conversation scenarios for testing.
    
    Returns:
        list: List of conversation scenarios, each containing multiple related queries
    """
    scenarios = [
        [
            "What is Braintrust and what are its main features?",
            "How do I get started with logging in Braintrust?", 
            "Can you show me an example of setting up an evaluation?"
        ],
        [
            "How does the Braintrust playground work?",
            "What are the different types of scorers available?",
            "How do I create a custom scorer for my use case?"
        ],
        [
            "What is the difference between datasets and experiments?",
            "How do I manage and version my datasets?",
            "Can I export my experiment results?"
        ],
        [
            "How does Braintrust handle model comparison?", 
            "What metrics should I look for when comparing models?",
            "How do I interpret the results of a model comparison?"
        ],
        [
            "What are assertions in Braintrust?",
            "How do I set up automated assertions for my evaluations?",
            "What happens when an assertion fails?"
        ]
    ]
    
    # Return a random sample of scenarios
    return random.sample(scenarios, min(num_scenarios, len(scenarios)))

def handle_tool_calls(tool_calls, parent_span) -> Dict[str, Any]:
    """Handle tool calls from OpenAI and return retrieved documents."""
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
                        "retrieved_documents": retrieved_documents
                    }
                )
                
                return {
                    "results": rag_response,
                    "tool_call_id": tool_call.id,
                    "retrieved_documents": retrieved_documents
                }

@traced
def process_single_query(query: str) -> str:
    """Process a single query using RAG and create an answer."""
    try:
        # Get the next model for this query
        current_model = get_next_model()
        print(f"   Using model: {current_model}")
        
        # Build messages for the conversation
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

        # Export the span for tool calls
        parent_span = current_span().export()
        
        # Handle tool calls and get documents
        retrieved_documents = []
        if response.choices[0].message.tool_calls:
            tool_response = handle_tool_calls(response.choices[0].message.tool_calls, parent_span)
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
            
            final_response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=1000
            )
            answer = final_response.choices[0].message.content
        else:
            # No tool calls - use the original response
            answer = response.choices[0].message.content
        
        # Log metadata for online scorers
        current_span().log(
            metadata={
                "retrieved_documents": retrieved_documents,
                "model": current_model,
                "provider": get_model_provider(current_model),
                "conversation_type": "single_turn"
            }
        )
        
        return answer or ""
        
    except Exception as e:
        return f"Error: {str(e)}"

@traced  
def process_multi_turn_conversation(conversation_queries: List[str]) -> str:
    """Process a multi-turn conversation using RAG."""
    try:
        # Get the next model for this conversation
        current_model = get_next_model()
        print(f"   Using model: {current_model}")
        
        # Build initial messages with system prompt
        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            }
        ]
        
        all_retrieved_documents = []
        
        # Process each query in the conversation
        for i, query in enumerate(conversation_queries):
            print(f"     Turn {i+1}: {query[:50]}...")
            
            # Add the user's query
            messages.append({
                "role": "user",
                "content": query
            })
            
            # Get completion with tool calls
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                tools=rag_tool,
                tool_choice="auto"
            )
            
            # Export the span for tool calls
            parent_span = current_span().export()
            
            # Handle tool calls if any
            if response.choices[0].message.tool_calls:
                tool_response = handle_tool_calls(response.choices[0].message.tool_calls, parent_span)
                all_retrieved_documents.extend(tool_response.get("retrieved_documents", []))
                
                # Convert ChatCompletionMessage to dict format for consistent handling
                assistant_message = response.choices[0].message
                assistant_content = getattr(assistant_message, 'content', '') or ''
                
                # For tool calls, content can be None/empty, so we need to handle this properly
                assistant_msg = {
                    "role": getattr(assistant_message, 'role', 'assistant'),
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                }
                
                # Only add content if it's not empty (OpenAI allows empty content for tool calls)
                if assistant_content:
                    assistant_msg["content"] = assistant_content
                
                messages.append(assistant_msg)
                
                # Add tool response
                messages.append({
                    "role": "tool",
                    "name": "get_documents", 
                    "content": json.dumps(tool_response["results"]),
                    "tool_call_id": tool_response["tool_call_id"]
                })
                
                final_response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    max_tokens=1000
                )
                assistant_response = getattr(final_response.choices[0].message, 'content', '') or ''
            else:
                # No tool calls - use the original response
                assistant_response = getattr(response.choices[0].message, 'content', '') or ''
            
            # Add assistant's response to conversation (only if not empty)
            if assistant_response and assistant_response.strip():
                messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
            else:
                # If assistant response is empty, add a fallback message
                messages.append({
                    "role": "assistant", 
                    "content": "I apologize, but I couldn't generate a response for that query."
                })
        
        # Log metadata for online scorers
        current_span().log(
            metadata={
                "retrieved_documents": all_retrieved_documents,
                "model": current_model,
                "provider": get_model_provider(current_model),
                "conversation_history": messages,
                "conversation_type": "multi_turn",
                "num_turns": len(conversation_queries)
            }
        )
        
        # Return the final assistant response
        assistant_messages = []
        for msg in messages:
            # Handle both dict format and ChatCompletionMessage objects
            if hasattr(msg, 'role'):  # ChatCompletionMessage object
                if getattr(msg, 'role', '') == 'assistant':
                    assistant_messages.append(msg)
            else:  # Dictionary format
                if msg.get("role") == "assistant":
                    assistant_messages.append(msg)
        
        if assistant_messages:
            final_message = assistant_messages[-1]
            # Extract content safely
            if hasattr(final_message, 'content'):
                return getattr(final_message, 'content', '') or ''
            else:
                return final_message.get("content", "")
        else:
            return ""
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main function to run RAG simulations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Simulation with Model Rotation and Multi-turn Support')
    parser.add_argument(
        '--logs-per-model', '-l',
        type=int,
        default=5,
        help='Number of logs to generate per model (default: 5)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['single', 'multi', 'both'],
        default='both',
        help='Simulation mode: single-turn, multi-turn, or both (default: both)'
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
    if args.logs_per_model < 1:
        print("Error: Number of logs per model must be at least 1")
        return
    
    if args.start_model < 0 or args.start_model >= len(MODELS_ROTATION):
        print(f"Error: Start model index must be between 0 and {len(MODELS_ROTATION)-1}")
        return
    
    # Set starting model position
    global model_counter
    model_counter = args.start_model
    
    print(f"🚀 Starting RAG Simulation")
    print(f"   Mode: {args.mode}")
    print(f"   Logs per model: {args.logs_per_model}")
    print(f"   Available models: {len(MODELS_ROTATION)}")
    print(f"   Models: {', '.join(MODELS_ROTATION)}")
    if args.start_model > 0:
        start_model_name = MODELS_ROTATION[args.start_model]
        print(f"   Starting from model index {args.start_model}: {start_model_name}")
    print()
    
    # Run single-turn simulations
    if args.mode in ['single', 'both']:
        print("=" * 60)
        print("SINGLE-TURN RAG SIMULATIONS")
        print("=" * 60)
        
        # Generate test queries
        total_single_queries = args.logs_per_model * len(MODELS_ROTATION)
        test_queries = generate_test_queries(total_single_queries)
        
        for i, query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] {query}")
            try:
                answer = process_single_query(query)
                print(f"  ✅ Answer generated successfully")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
            print()
    
    # Run multi-turn simulations  
    if args.mode in ['multi', 'both']:
        print("=" * 60)
        print("MULTI-TURN RAG SIMULATIONS") 
        print("=" * 60)
        
        # Create multi-turn scenarios
        num_scenarios = args.logs_per_model * len(MODELS_ROTATION) // 3  # Fewer scenarios, but each has multiple turns
        multi_turn_scenarios = create_multi_turn_scenarios(max(num_scenarios, 2))
        
        for i, scenario in enumerate(multi_turn_scenarios, 1):
            print(f"[Conversation {i}/{len(multi_turn_scenarios)}] Multi-turn scenario with {len(scenario)} turns")
            try:
                final_response = process_multi_turn_conversation(scenario)
                print(f"  ✅ Multi-turn conversation completed successfully")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                continue
            print()
    
    print("✅ RAG Simulation completed successfully!")
    print(f"Check Braintrust dashboard for detailed traces and online scoring results.")
    print(f"Total models used: {len(MODELS_ROTATION)}")

if __name__ == "__main__":
    main()