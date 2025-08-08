import os
import json
import datetime
from braintrust import Eval, init_dataset, traced, wrap_openai
from dotenv import load_dotenv
import sys
from openai import OpenAI

# Add the src directory to the path so we can import from rag_braintrust_bot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress httpx logs that come from OpenAI client calls to Braintrust proxy
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)

from rag_braintrust_bot.rag_demo import rag_tool, SYSTEM_PROMPT
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents

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
            
            return assistant_response or ""
            
        except Exception as e:
            print(f"Error in task execution with {model_name}: {str(e)}")
            hooks.metadata['error'] = str(e)
            return ""
    
    return task  # Return the configured task function

# === SCORING FUNCTIONS ===
# These functions evaluate different aspects of the RAG system's responses

def document_retrieval_check(input, output, metadata):
    """
    Check if documents were retrieved and assess their quality.
    
    Args:
        input: The input data
        output: The generated response
        metadata: Contains retrieved document information
    
    Returns:
        dict: Score and metadata about document retrieval accuracy
    """
    # Extract retrieved documents
    retrieved_docs = metadata.get('retrieved_documents', [])
    
    # Check if we got any documents
    if not retrieved_docs:
        return {
            'score': 0,
            'name': 'document_retrieval_check',
            'metadata': {
                'retrieved_docs': [],
                'doc_count': 0,
                'reasoning': 'No documents were retrieved'
            }
        }
    
    # Score based on document count and relevance scores
    doc_count = len(retrieved_docs)
    avg_score = sum(doc.get('score', 0) for doc in retrieved_docs) / doc_count if doc_count > 0 else 0
    
    # Combine document count (normalized) and average relevance score
    count_score = min(doc_count / 3, 1.0)  # Normalize to 1 for 3+ docs
    final_score = (count_score * 0.3) + (avg_score * 0.7)
    
    return {
        'score': final_score,
        'name': 'document_retrieval_check',
        'metadata': {
            'retrieved_docs': retrieved_docs,
            'doc_count': doc_count,
            'avg_relevance_score': avg_score,
            'reasoning': f'Retrieved {doc_count} documents with avg relevance {avg_score:.3f}'
        }
    }

# Simplified scoring functions for single-turn with chat history context

def answer_relevance_check(input, output, metadata):
    """
    Check if the generated answers are relevant to the queries in multi-turn context.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Additional metadata
    
    Returns:
        dict: Score and metadata about answer relevance
    """
    # Get the final response
    if isinstance(output, dict):
        final_response = output.get('final_response', '')
    else:
        final_response = str(output)
    
    # Get the last user query
    conversation_history = metadata.get('conversation_history', [])
    last_user_query = ''
    for msg in reversed(conversation_history):
        if msg.get('role') == 'user':
            last_user_query = msg.get('content', '')
            break
    
    if not final_response or not last_user_query:
        return {
            'score': 0,
            'name': 'answer_relevance_check',
            'metadata': {
                'reasoning': 'Empty response or no user query found'
            }
        }
    
    # Simple keyword overlap scoring
    query_words = set(last_user_query.lower().split())
    response_words = set(final_response.lower().split())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    query_words = query_words - stopwords
    response_words = response_words - stopwords
    
    if not query_words:
        overlap_score = 0.5  # Neutral score if no meaningful query words
    else:
        overlap = len(query_words.intersection(response_words))
        overlap_score = overlap / len(query_words)
    
    # Length check - penalize very short responses
    length_score = min(len(final_response) / 100, 1.0)  # Normalize to 1 for 100+ chars
    
    # Combine scores
    final_score = (overlap_score * 0.7) + (length_score * 0.3)
    
    return {
        'score': final_score,
        'name': 'answer_relevance_check',
        'metadata': {
            'query_words': list(query_words)[:20],  # Limit for readability
            'overlap_count': len(query_words.intersection(response_words)),
            'overlap_score': overlap_score,
            'length_score': length_score,
            'response_length': len(final_response)
        }
    }

def answer_faithfulness_check(input, output, metadata):
    """
    Check if the answers are faithful to retrieved documents across all turns.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Contains retrieved document information
    
    Returns:
        dict: Score and metadata about answer faithfulness
    """
    # Get all responses from conversation
    conversation_history = metadata.get('conversation_history', [])
    assistant_responses = [msg.get('content', '') for msg in conversation_history if msg.get('role') == 'assistant']
    
    # Get all retrieved documents
    all_retrieved_docs = metadata.get('all_retrieved_documents', [])
    
    if not assistant_responses or not all_retrieved_docs:
        return {
            'score': 0.5,  # Neutral score if no context available
            'name': 'answer_faithfulness_check',
            'metadata': {
                'reasoning': 'No responses or retrieved documents available for faithfulness check'
            }
        }
    
    # Combine all document content
    document_contents = []
    for doc in all_retrieved_docs:
        if 'content' in doc:
            document_contents.append(doc['content'])
    
    context = ' '.join(document_contents)
    all_responses = ' '.join(assistant_responses)
    
    # Simple faithfulness check based on content overlap
    context_words = set(context.lower().split())
    response_words = set(all_responses.lower().split())
    
    # Remove stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    context_words = context_words - stopwords
    response_words = response_words - stopwords
    
    if not response_words:
        return {
            'score': 0,
            'name': 'answer_faithfulness_check',
            'metadata': {
                'reasoning': 'Empty responses'
            }
        }
    
    # Calculate what percentage of response words are supported by context
    supported_words = response_words.intersection(context_words)
    faithfulness_score = len(supported_words) / len(response_words)
    
    return {
        'score': faithfulness_score,
        'name': 'answer_faithfulness_check',
        'metadata': {
            'total_response_words': len(response_words),
            'supported_words': len(supported_words),
            'faithfulness_score': faithfulness_score,
            'document_count': len(all_retrieved_docs),
            'unsupported_words': list(response_words - context_words)[:10]  # First 10 unsupported words
        }
    }

def response_structure_check(input, output, metadata):
    """
    Check if responses maintain good structure throughout the conversation.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Additional metadata
    
    Returns:
        dict: Score and metadata about response structure
    """
    conversation_history = metadata.get('conversation_history', [])
    assistant_responses = [msg.get('content', '') for msg in conversation_history if msg.get('role') == 'assistant']
    
    if not assistant_responses:
        return {
            'score': 0,
            'name': 'response_structure_check',
            'metadata': {
                'reasoning': 'No assistant responses found'
            }
        }
    
    total_score = 0
    all_checks = []
    
    # Evaluate structure for each response
    for idx, response in enumerate(assistant_responses):
        score = 0
        checks = {}
        
        # Check for adequate length (not too short)
        if len(response) > 50:
            score += 0.2
            checks['adequate_length'] = True
        else:
            checks['adequate_length'] = False
        
        # Check for structured formatting (bullet points, numbers, etc.)
        if any(marker in response for marker in ['"', '-', '*', '1.', '2.', '3.']):
            score += 0.2
            checks['has_formatting'] = True
        else:
            checks['has_formatting'] = False
        
        # Check for code examples (backticks)
        if '`' in response:
            score += 0.2
            checks['has_code_examples'] = True
        else:
            checks['has_code_examples'] = False
        
        # Check for clear sentence structure (periods)
        if response.count('.') >= 2:
            score += 0.2
            checks['clear_sentences'] = True
        else:
            checks['clear_sentences'] = False
        
        # Check for helpful details (mentions specific features/tools)
        braintrust_terms = ['braintrust', 'evaluation', 'dataset', 'prompt', 'logging', 'tracing', 'playground', 'experiment', 'score']
        if any(term in response.lower() for term in braintrust_terms):
            score += 0.2
            checks['mentions_braintrust_features'] = True
        else:
            checks['mentions_braintrust_features'] = False
        
        total_score += score
        all_checks.append({
            'turn': idx,
            'score': score,
            'checks': checks
        })
    
    # Average score across all responses
    avg_score = total_score / len(assistant_responses)
    
    # Check for consistency in structure quality
    scores = [check['score'] for check in all_checks]
    if len(scores) > 1:
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        consistency_bonus = 0.1 if score_variance < 0.1 else 0  # Bonus for consistent quality
    else:
        consistency_bonus = 0
    
    final_score = min(avg_score + consistency_bonus, 1.0)
    
    return {
        'score': final_score,
        'name': 'response_structure_check',
        'metadata': {
            'average_structure_score': avg_score,
            'response_count': len(assistant_responses),
            'per_turn_checks': all_checks[:3],  # Show first 3 turns for brevity
            'reasoning': f'Average structure score: {avg_score:.2f} across {len(assistant_responses)} responses'
        }
    }

# === MAIN EXECUTION ===

# Load the dataset once (it will be reused for all models)
dataset = init_dataset(project=project_name, name="BraintrustMultiTurnQA")

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
            name="not_eval",
            task=task_for_model,
            data=dataset,
            scores=[
                document_retrieval_check,
                answer_relevance_check,
                answer_faithfulness_check,
                response_structure_check
            ],
            metadata={
                'model': model_name,
                'dataset': 'BraintrustMultiTurnQA',
            },
            experiment_name=f"multiturn_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    except Exception as e:
        print(f"\nError running evaluation for {model_name}: {str(e)}")
        print(f"Skipping {model_name} and continuing with next model...\n")
        continue

print(f"\n{'='*60}")
print("All model evaluations completed!")
print(f"{'='*60}\n")