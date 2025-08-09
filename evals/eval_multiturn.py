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

from rag_braintrust_bot.rag_simulation import rag_tool, SYSTEM_PROMPT
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents

# Import local scorer functions
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
    #"o3-mini",
    #"gpt-5-mini",
    #"claude-3-5-sonnet-latest",
    #"claude-4-sonnet-20250514",
    # Add more models as needed
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
        Run the RAG system with multi-turn conversation support using live LLM calls.
        
        This function handles the Braintrust dataset format where each item contains
        a full conversation history. It extracts the last user message to process
        and uses all previous messages as context.
        
        Args:
            input: Can be:
                - List of messages (from Braintrust dataset) - full conversation
                - Dict with 'input' and 'chat_history' keys - structured format
                - String - simple single query
        
        Returns:
            str: The assistant's response to the last user message
        """
        # Store which model is being used
        hooks.metadata['model'] = model_name
        
        try:
            # === STEP 1: Parse the input to extract user query and chat history ===
            
            if isinstance(input, list):
                # BRAINTRUST DATASET FORMAT: Full conversation as a list
                # Example: [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', ...}, {'role': 'user', 'content': 'Question?'}]
                # We need to find the LAST user message and treat everything before it as context
                
                user_input = ''
                chat_history = []
                
                # Walk backwards through the conversation to find the most recent user message
                last_user_idx = -1
                for i in range(len(input) - 1, -1, -1):
                    if input[i].get('role') == 'user':
                        last_user_idx = i
                        user_input = input[i].get('content', '')
                        break
                
                # Everything before the last user message becomes chat history
                if last_user_idx > 0:
                    chat_history = input[:last_user_idx]
                
            elif isinstance(input, dict) and 'input' in input:
                # STRUCTURED FORMAT: Separate current query and history
                # Used when we manually create evaluation data
                user_input = input.get('input', '')
                chat_history = input.get('chat_history', [])
            else:
                # SIMPLE FORMAT: Just a string query with no history
                user_input = str(input) if input else ''
                chat_history = []
            
            # Store metadata for debugging
            hooks.metadata['user_input'] = user_input
            hooks.metadata['chat_history_length'] = len(chat_history)
            
            # === STEP 2: Build the messages array for OpenAI API ===
            
            messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
            ]
            
            # Add the conversation history
            # IMPORTANT: Tool messages in the dataset have 'content' as a dict,
            # but OpenAI API expects 'content' to be a JSON string for tool messages
            for msg in chat_history:
                if msg.get('role') == 'tool' and isinstance(msg.get('content'), dict):
                    # Convert tool response dict to JSON string
                    messages.append({
                        **msg,  # Copy all fields (role, tool_call_id, etc.)
                        'content': json.dumps(msg['content'])  # Convert content dict to string
                    })
                else:
                    # Regular messages (user/assistant) can be added as-is
                    messages.append(msg)
            
            # Add the current user's question
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # === STEP 3: Make the LLM call with RAG tool available ===
            
            response = client.chat.completions.create(
                model=model_name,  # Use the model passed in
            messages=messages,
            tools=rag_tool,  # The RAG tool for document retrieval
            tool_choice="auto"  # Let the model decide if it needs to search
            )
            
            # === STEP 4: Handle tool calls (if the model wants to search for documents) ===
            
            all_retrieved_docs = []
            if response.choices[0].message.tool_calls:
                # Add the assistant's tool call message
                messages.append(response.choices[0].message)
                
                # Process each tool call
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "get_documents":
                        # Parse arguments and call the RAG tool
                        args = json.loads(tool_call.function.arguments)
                        query_for_docs = args.get("query", "")
                        
                        # Call the RAG tool directly (this will be traced as a separate operation)
                        rag_response = get_documents(query_for_docs)
                        all_retrieved_docs.extend(rag_response["documents"])
                        
                        # Add tool response
                        messages.append({
                            "role": "tool",
                            "name": "get_documents",
                            "content": json.dumps(rag_response),
                            "tool_call_id": tool_call.id
                        })
            
                # Get final response after tool calls
                final_response = client.chat.completions.create(
                    model=model_name,  # Use the model passed in
                messages=messages,
                max_tokens=1000
                )
                
                assistant_response = final_response.choices[0].message.content
            else:
                # No tool calls, use the original response
                assistant_response = response.choices[0].message.content
            
            # Store metadata for scoring
            hooks.metadata['retrieved_documents'] = all_retrieved_docs
            hooks.metadata['tool_calls_made'] = len(response.choices[0].message.tool_calls) if response.choices[0].message.tool_calls else 0
            hooks.metadata['conversation_history'] = messages  # Full conversation for context-aware scorers
            hooks.metadata['context_type'] = 'multi_turn'  # Signal for context-aware scorers
            
            return assistant_response or ""
            
        except Exception as e:
            print(f"Error in task execution with {model_name}: {str(e)}")
            hooks.metadata['error'] = str(e)
            return ""
    
    return task  # Return the configured task function

# === MAIN EXECUTION ===

# Load the dataset once (it will be reused for all models)
dataset = init_dataset(project=project_name, name="BraintrustMultiTurnQA")

# Load the LLM judges that were pushed to Braintrust
rag_factuality_llm = init_function(project_name=project_name, slug="rag-factuality-llm")
rag_relevance_llm = init_function(project_name=project_name, slug="rag-relevance-llm")
rag_completeness_llm = init_function(project_name=project_name, slug="rag-completeness-llm")

# Run evaluation for each model
for model_name in MODELS_TO_EVALUATE:
    print(f"\n{'='*60}")
    print(f"Running evaluation for model: {model_name}")
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
            metadata={
                'model': model_name,
                'dataset': 'BraintrustMultiTurnQA',
            },
            experiment_name=f"multiturn_{model_name}_{datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    except Exception as e:
        print(f"\nError running evaluation for {model_name}: {str(e)}")
        print(f"Skipping {model_name} and continuing with next model...\n")
        continue

print(f"\n{'='*60}")
print("All multiturn evaluations completed!")
print(f"{'='*60}\n")