import os
import braintrust
from braintrust import Span, Attachment, traced, wrap_openai, init_logger
from dotenv import load_dotenv
from typing import List, Dict, Any
import openai
import json
from tools.retrieval_tool import handler as get_documents
import sys

# Load environment variables
load_dotenv('.env.local')

# Validate required environment variables
required_env_vars = [
    "BRAINTRUST_API_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please ensure these are set in your .env.local file")
    sys.exit(1)

logger = init_logger(project="rag-braintrust-bot")

# Initialize wrapped OpenAI client
client = wrap_openai(
    openai.OpenAI(
        base_url="https://braintrustproxy.com/v1",
        api_key=os.getenv('BRAINTRUST_API_KEY')
    )
)

# Sample queries to test the system
test_queries = [
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

def handle_tool_calls(tool_calls):
    """Handle tool calls from OpenAI."""
    results = []
    for tool_call in tool_calls:
        if tool_call.function.name == "get_documents":
            args = json.loads(tool_call.function.arguments)
            documents = get_documents(args["query"])
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "get_documents",
                "content": json.dumps(documents)
            })
    return results

@traced
def process_query(query: str):
    """Process a query using RAG and create an answer."""
    print(f"\nProcessing query: {query}")
    
    try:
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
            model="gpt-4",
            messages=messages,
            tools=rag_tool,
            tool_choice="auto"
        )
        
        # Handle tool calls and get documents
        tool_results = handle_tool_calls(response.choices[0].message.tool_calls)
        
        # Add tool results and get final answer
        messages.extend([
            response.choices[0].message,
            *tool_results
        ])
        
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000
        )
        
        answer = final_response.choices[0].message.content
        
        print(f"Created answer using retrieved documents")
        print("\nAnswer:", answer)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise

def main():
    """Main function to process all test queries."""
    print(f"Starting RAG + LLM evaluation with {len(test_queries)} test queries")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}/{len(test_queries)}")
        try:
            process_query(query)
        except Exception as e:
            print(f"Failed to process query: {str(e)}")
            continue

if __name__ == "__main__":
    main() 