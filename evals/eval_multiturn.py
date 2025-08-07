import os
import json
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

# Create wrapped OpenAI client for automatic tracing
client = wrap_openai(
    OpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",
        api_key=os.getenv('BRAINTRUST_API_KEY', '')
    )
)

@traced
def task(input, hooks):
    """
    Run the RAG system with multi-turn conversation support using live LLM calls.
    Follows the pattern of extracting input and chat_history from the dataset.
    
    Args:
        input: Dictionary containing:
            - input: The current user query (string) 
            - chat_history: Previous conversation turns (list)
    
    Returns:
        str: The assistant's response
    """
    try:
        # Handle different input formats from the dataset
        if isinstance(input, list):
            # Input is the full conversation from Braintrust dataset
            # Find the last user message and use everything before it as chat history
            user_input = ''
            chat_history = []
            
            # Find the last user message index
            last_user_idx = -1
            for i in range(len(input) - 1, -1, -1):
                if input[i].get('role') == 'user':
                    last_user_idx = i
                    user_input = input[i].get('content', '')
                    break
            
            # Everything before the last user message is chat history
            if last_user_idx > 0:
                chat_history = input[:last_user_idx]
            
        elif isinstance(input, dict) and 'input' in input:
            # Structured input with separate input and chat_history
            user_input = input.get('input', '')
            chat_history = input.get('chat_history', [])
        elif isinstance(input, dict):
            # Single dict that might be a message
            user_input = input.get('content', '')
            chat_history = []
        else:
            # Fallback for simple string input
            user_input = str(input)
            chat_history = []
        
        hooks.metadata['user_input'] = user_input
        hooks.metadata['chat_history_length'] = len(chat_history)
        
        # Build the conversation messages
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        
        # Add chat history, converting tool message content to strings if needed
        for msg in chat_history:
            if msg.get('role') == 'tool' and isinstance(msg.get('content'), dict):
                # Tool messages need content as a JSON string
                messages.append({
                    **msg,
                    'content': json.dumps(msg['content'])
                })
            else:
                messages.append(msg)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Make the LLM call with tools (wrapOpenAI will automatically trace this)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=rag_tool,
            tool_choice="auto"
        )
        
        # Handle tool calls
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
                model="gpt-4o-mini",
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
        print(f"Error in task execution: {str(e)}")
        hooks.metadata['error'] = str(e)
        return ""

# Remove unused functions - we're now using live calls with the main task function

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

@traced
def _extract_documents_from_turn(tool_content):
    """Extract documents from tool response content."""
    documents = []
    if isinstance(tool_content, dict) and "documents" in tool_content:
        documents = tool_content["documents"]
    elif isinstance(tool_content, str):
        try:
            parsed_content = json.loads(tool_content)
            if "documents" in parsed_content:
                documents = parsed_content["documents"]
        except json.JSONDecodeError:
            pass
    return documents

@traced
def _process_conversation_turn(turn, turn_count, all_retrieved_docs, conversation_history, hooks):
    """Process a single conversation turn and update metadata."""
    role = turn.get("role", "")
    
    if role == "user":
        turn_count += 1
        conversation_history.append(turn)
        hooks.metadata[f'turn_{turn_count-1}_query'] = turn.get("content", "")
        
    elif role == "assistant":
        conversation_history.append(turn)
        
        # Check if this assistant message has tool calls
        if "tool_calls" in turn:
            hooks.metadata[f'turn_{turn_count-1}_has_tool_calls'] = True
        
    elif role == "tool":
        # Extract documents from tool response
        tool_content = turn.get("content", {})
        documents = _extract_documents_from_turn(tool_content)
        if documents:
            all_retrieved_docs.extend(documents)
            hooks.metadata[f'turn_{turn_count-1}_retrieved_docs'] = documents
    
    return turn_count

@traced
def _run_prerecorded_conversation(turns, hooks):
    """Process pre-recorded conversation from dataset."""
    all_retrieved_docs = []
    conversation_history = []
    turn_count = 0
    
    # Process each turn to extract information
    for turn_idx, turn in enumerate(turns):
        turn_count = _process_conversation_turn(
            turn, turn_count, all_retrieved_docs, conversation_history, hooks
        )
    
    # Store overall conversation metadata
    hooks.metadata['all_retrieved_documents'] = all_retrieved_docs
    hooks.metadata['conversation_history'] = conversation_history
    hooks.metadata['turn_count'] = turn_count
    hooks.metadata['evaluation_mode'] = 'prerecorded'
    
    # Get the last assistant response as the final response
    last_assistant_response = ""
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant" and turn.get("content"):
            last_assistant_response = turn.get("content", "")
            break
    
    return {
        "final_response": last_assistant_response,
        "conversation_history": conversation_history
    }

def load_multiturn_conversations():
    """
    Load multi-turn conversations from the JSON dataset.
    Returns the conversations as-is for direct evaluation.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets', 'braintrust_multiturn_qa_conversations.json')
    
    with open(dataset_path, 'r') as f:
        conversations = json.load(f)
    
    # Transform each conversation into individual evaluation cases
    eval_cases = []
    
    for conversation in conversations:
        turns = conversation.get('turns', [])
        chat_history = []
        
        for i, turn in enumerate(turns):
            if turn.get('role') == 'user':
                # Create evaluation case for this user turn
                eval_case = {
                    'input': {
                        'input': turn.get('content', ''),
                        'chat_history': chat_history.copy()
                    },
                    'expected': None,
                    'metadata': {
                        'conversation_id': conversation.get('id'),
                        'turn_index': i,
                        'description': conversation.get('description', '')
                    }
                }
                eval_cases.append(eval_case)
            
            # Add this turn to chat history
            chat_history.append(turn)
    
    return eval_cases

# Run the evaluation with the Braintrust dataset
Eval(
    project_name,
    task=task,
    data=init_dataset(project=project_name, name="BraintrustMultiTurnQA"),
    scores=[
        document_retrieval_check,
        answer_relevance_check,
        answer_faithfulness_check,
        response_structure_check
    ]
)