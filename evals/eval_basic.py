import os
import json
import datetime
from braintrust import Eval, init_dataset, traced, wrap_openai, init_function
from dotenv import load_dotenv
import sys
from openai import OpenAI
import pytz

# Add the src directory to the path so we can import from rag_braintrust_bot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress httpx logs that come from OpenAI client calls to Braintrust proxy
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)

from rag_braintrust_bot.rag_demo import rag_tool, SYSTEM_PROMPT
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents

# Import local context-aware scorer functions
from push_scorers import (
    rag_precision,
    rag_recall,
    rag_f1
)

# Load environment variables
load_dotenv()

project_name = os.getenv('BRAINTRUST_PROJECT_NAME', 'rag-braintrust-bot')

# List of models to evaluate
MODELS_TO_EVALUATE = [
    "gpt-4o-mini",
    # Add more models as needed when they become available
    # "gpt-4o",
    # "claude-3-5-sonnet-20241022",
]

# Create wrapped OpenAI client for automatic tracing
client = wrap_openai(
    OpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",
        api_key=os.getenv('BRAINTRUST_API_KEY', '')
    )
)

def create_task(model_name):
    """
    Create a task function for a specific model.
    
    Args:
        model_name: The name of the model to use for evaluation
    
    Returns:
        A task function configured for the specified model
    """
    @traced
    def task(input, hooks):
        """
        Run the RAG system and gather results for evaluation.
        
        Args:
            input_data: Dictionary or string containing the query and any metadata
            hooks: Braintrust hooks for logging metadata
        
        Returns:
            str: The final response from the RAG system
        """
        # Store which model is being used
        hooks.metadata['model'] = model_name
        
        # Handle both string and dictionary input formats
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict):
            query = input.get('query', input.get('input', ''))
        else:
            query = str(input)
        
        try:
            # Create the conversation without complex span handling
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
                model=model_name,  # Use the model passed in
                messages=messages,
                tools=rag_tool,
                tool_choice="auto"
            )
            
            # Handle tool calls without complex span hierarchies
            tool_responses = []
            retrieved_docs = []
            
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "get_documents":
                        # Parse the query from the function arguments
                        args = json.loads(tool_call.function.arguments)
                        query_for_docs = args.get("query", "")
                        
                        # Call the RAG tool directly
                        rag_response = get_documents(query_for_docs)
                        
                        # Store document info for scoring
                        retrieved_docs = rag_response["documents"]
                        
                        tool_responses.append({
                            "role": "tool",
                            "name": "get_documents",
                            "content": json.dumps(rag_response),
                            "tool_call_id": tool_call.id
                        })
            
            # Add tool responses and get final answer if we had tool calls
            if tool_responses:
                messages.extend([response.choices[0].message] + tool_responses)
                
                final_response = client.chat.completions.create(
                    model=model_name,  # Use the model passed in
                    messages=messages,
                    max_tokens=1000
                )
                
                answer = final_response.choices[0].message.content
            else:
                # No tool calls, use the original response
                answer = response.choices[0].message.content
            
            # Store retrieval information in metadata for scoring
            hooks.metadata['retrieved_documents'] = retrieved_docs
            hooks.metadata['query'] = query
            hooks.metadata['context_type'] = 'single_turn'  # Signal for context-aware scorers
            
            return answer or ""
            
        except Exception as e:
            print(f"Error in task execution with {model_name}: {str(e)}")
            hooks.metadata['error'] = str(e)
            return ""
    
    return task  # Return the configured task function

# Local scorer functions removed - now using Braintrust-pushed scorers




# === MAIN EXECUTION ===

# Load the dataset once (it will be reused for all models)
dataset = init_dataset(project=project_name, name="BraintrustDocumentationQA")

# Load the LLM judges that were pushed to Braintrust
rag_factuality_llm = init_function(project_name=project_name, slug="rag-factuality-llm")
rag_relevance_llm = init_function(project_name=project_name, slug="rag-relevance-llm")
rag_completeness_llm = init_function(project_name=project_name, slug="rag-completeness-llm")

# Run evaluation for each model
for model_name in MODELS_TO_EVALUATE:
    print(f"\n{'='*60}")
    print(f"Running basic evaluation for model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        # Create a task function configured for this specific model
        task_for_model = create_task(model_name)
        
        # Run the evaluation with the model name in the experiment name
        Eval(
            project_name,
            task=task_for_model,
            data=dataset,
            scores=[
                # Local context-aware scorers only
                rag_precision,
                rag_recall,
                rag_f1,
                rag_factuality_llm,
                rag_relevance_llm,
                rag_completeness_llm,
            ],
            experiment_name=f"basic_{model_name}_{datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    except Exception as e:
        print(f"\nError running evaluation for {model_name}: {str(e)}")
        print(f"Skipping {model_name} and continuing with next model...\n")
        continue

print(f"\n{'='*60}")
print("All basic model evaluations completed!")
print(f"{'='*60}\n")