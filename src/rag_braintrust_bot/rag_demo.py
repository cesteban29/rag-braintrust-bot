import os
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
        
        # Generate questions using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

# Generate dynamic test queries
test_queries = generate_test_queries()

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
    for tool_call in tool_calls:
        if tool_call.function.name == "get_documents":
            # Parse the query from the function arguments
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            
            # Create a span for document retrieval
            with logger.start_span(name="get_documents", type="tool", parent=parent_span) as span:
                # Call the RAG tool
                rag_response = get_documents(query)
                
                # Log the input query & document ids
                span.log(
                    inputs={"query": query},
                    metadata={"tool_name": "get_documents", "document_ids": [doc['id'] for doc in rag_response["documents"]]}
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
                    "tool_call_id": tool_call.id
                }

@traced
def process_query(query: str):
    """Process a query using RAG and create an answer."""
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
            model="gpt-4o-mini",
            messages=messages,
            tools=rag_tool,
            tool_choice="auto"
        )

        # export the span slug
        parent_span = current_span().export()
        
        # Handle tool calls and get documents
        tool_response = handle_tool_calls(response.choices[0].message.tool_calls, parent_span)
        
        # Add tool results and get final answer
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
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )
        
        answer = final_response.choices[0].message.content
        
        # Return the answer so Braintrust can log it as the span output
        return answer
        
    except Exception as e:
        # Return empty string on error so Braintrust still has an output
        return f"Error: {str(e)}"

def main():
    """Main function to process all test queries."""
    print(f"Running RAG demo with {len(test_queries)} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query}")
        try:
            answer = process_query(query)
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    print("\n✅ Demo completed. Check Braintrust logs for detailed traces.")

if __name__ == "__main__":
    main() 