import os
import json
from braintrust import Eval, init_dataset, traced
from dotenv import load_dotenv
import sys

# Add the src directory to the path so we can import from rag_braintrust_bot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress httpx logs that come from OpenAI client calls to Braintrust proxy
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)

from rag_braintrust_bot.rag_demo import client, rag_tool, SYSTEM_PROMPT
from rag_braintrust_bot.tools.retrieval_tool import handler as get_documents
import json

# Load environment variables
load_dotenv()

project_name = os.getenv('BRAINTRUST_PROJECT_NAME', 'rag-braintrust-bot')

@traced
def task(input, hooks):
    """
    Run the RAG system with multi-turn conversation support.
    Can work with either live RAG calls or pre-recorded conversations from dataset.
    
    Args:
        input: Dictionary containing conversation turns or a single query
        hooks: Braintrust hooks for logging metadata
    
    Returns:
        dict: The conversation history and final response
    """
    # Handle input format for multi-turn conversations
    if isinstance(input, str):
        # Single query - convert to single turn
        turns = [{"role": "user", "content": input}]
        use_live_rag = True
    elif isinstance(input, dict):
        if 'turns' in input:
            # Multi-turn conversation format from dataset
            turns = input['turns']
            use_live_rag = False  # Use pre-recorded conversation
        elif 'messages' in input:
            # Alternative message format
            turns = input['messages']
            use_live_rag = True
        else:
            # Single query in dict format
            query = input.get('query', input.get('input', ''))
            turns = [{"role": "user", "content": query}]
            use_live_rag = True
    else:
        turns = [{"role": "user", "content": str(input)}]
        use_live_rag = True
    
    try:
        if use_live_rag:
            # Live RAG evaluation - make actual LLM calls
            result = _run_live_rag_conversation(turns, hooks)
        else:
            # Pre-recorded conversation evaluation - use dataset conversation
            result = _run_prerecorded_conversation(turns, hooks)
        
        # Return just the final response for Braintrust logging
        return result.get("final_response", "")
        
    except Exception as e:
        print(f"Error in multi-turn task execution: {str(e)}")
        hooks.metadata['error'] = str(e)
        return ""

@traced
def _run_live_rag_conversation(turns, hooks):
    """Run live RAG conversation with actual LLM calls."""
    # Initialize conversation with system prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]
    
    # Store all retrieved documents across turns
    all_retrieved_docs = []
    conversation_history = []
    
    # Process each turn in the conversation
    for turn_idx, turn in enumerate(turns):
        if turn.get("role") == "user":
            # Add user message
            messages.append(turn)
            conversation_history.append(turn)
            
            # Get response with tool calls
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=rag_tool,
                tool_choice="auto"
            )
            
            # Handle tool calls for this turn
            tool_responses = []
            turn_retrieved_docs = []
            
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "get_documents":
                        # Parse the query from the function arguments
                        args = json.loads(tool_call.function.arguments)
                        query_for_docs = args.get("query", "")
                        
                        # Call the RAG tool
                        rag_response = get_documents(query_for_docs)
                        
                        # Store document info
                        turn_retrieved_docs = rag_response["documents"]
                        all_retrieved_docs.extend(turn_retrieved_docs)
                        
                        tool_responses.append({
                            "role": "tool",
                            "name": "get_documents",
                            "content": json.dumps(rag_response),
                            "tool_call_id": tool_call.id
                        })
            
            # Add assistant's tool call message and tool responses
            messages.append(response.choices[0].message)
            
            # Get final answer if we had tool calls
            if tool_responses:
                messages.extend(tool_responses)
                
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1000
                )
                
                assistant_message = {
                    "role": "assistant",
                    "content": final_response.choices[0].message.content
                }
            else:
                # No tool calls, use the original response
                assistant_message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }
            
            # Add assistant response to messages and history
            messages.append(assistant_message)
            conversation_history.append(assistant_message)
            
            # Store turn metadata
            hooks.metadata[f'turn_{turn_idx}_retrieved_docs'] = turn_retrieved_docs
            hooks.metadata[f'turn_{turn_idx}_query'] = turn.get("content", "")
        
        elif turn.get("role") == "assistant":
            # Pre-existing assistant message (for context)
            messages.append(turn)
            conversation_history.append(turn)
    
    # Store overall conversation metadata
    hooks.metadata['all_retrieved_documents'] = all_retrieved_docs
    hooks.metadata['conversation_history'] = conversation_history
    hooks.metadata['turn_count'] = len([t for t in turns if t.get("role") == "user"])
    
    # Return the full conversation or just the last response
    if len(conversation_history) > 0:
        last_response = conversation_history[-1].get("content", "")
    else:
        last_response = ""
    
    return {
        "final_response": last_response,
        "conversation_history": conversation_history
    }

def document_retrieval_check(input, output, metadata):
    """
    Check if the correct documents were retrieved across all turns.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Contains tool call information
    
    Returns:
        dict: Score and metadata about document retrieval accuracy
    """
    # Extract all retrieved documents
    all_retrieved_docs = metadata.get('all_retrieved_documents', [])
    
    # Check if we got any documents
    if not all_retrieved_docs:
        return {
            'score': 0,
            'name': 'document_retrieval_check',
            'metadata': {
                'retrieved_docs': [],
                'total_doc_count': 0,
                'reasoning': 'No documents were retrieved in any turn'
            }
        }
    
    # Score based on document count and relevance scores
    doc_count = len(all_retrieved_docs)
    avg_score = sum(doc.get('score', 0) for doc in all_retrieved_docs) / doc_count if doc_count > 0 else 0
    
    # Check document diversity (unique documents)
    unique_docs = len(set(doc.get('id', doc.get('content', '')[:50]) for doc in all_retrieved_docs))
    diversity_score = unique_docs / doc_count if doc_count > 0 else 0
    
    # Combine metrics
    count_score = min(doc_count / 5, 1.0)  # Normalize to 1 for 5+ docs across conversation
    final_score = (count_score * 0.3) + (avg_score * 0.5) + (diversity_score * 0.2)
    
    return {
        'score': final_score,
        'name': 'document_retrieval_check',
        'metadata': {
            'total_doc_count': doc_count,
            'unique_doc_count': unique_docs,
            'avg_relevance_score': avg_score,
            'diversity_score': diversity_score,
            'reasoning': f'Retrieved {doc_count} documents ({unique_docs} unique) with avg relevance {avg_score:.3f}'
        }
    }

def conversation_coherence_check(input, output, metadata):
    """
    Check if the conversation maintains coherence across turns.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Contains conversation history
    
    Returns:
        dict: Score and metadata about conversation coherence
    """
    conversation_history = metadata.get('conversation_history', [])
    
    if len(conversation_history) < 2:
        return {
            'score': 1.0,  # Single turn is inherently coherent
            'name': 'conversation_coherence_check',
            'metadata': {
                'reasoning': 'Single turn conversation - coherence not applicable'
            }
        }
    
    score = 0
    checks = {}
    
    # Check for topic consistency
    all_messages = ' '.join([msg.get('content', '') for msg in conversation_history])
    
    # Extract key terms from first user message
    first_user_msg = next((msg for msg in conversation_history if msg.get('role') == 'user'), {})
    first_content = first_user_msg.get('content', '').lower()
    
    # Simple keyword tracking for topic consistency
    key_terms = set(first_content.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'who'}
    
    # Check if key terms appear in subsequent responses
    subsequent_messages = conversation_history[2:]  # Skip first user and assistant messages
    if subsequent_messages and key_terms:
        term_continuity = 0
        for msg in subsequent_messages:
            content = msg.get('content', '').lower()
            matching_terms = sum(1 for term in key_terms if term in content)
            term_continuity += matching_terms / len(key_terms) if key_terms else 0
        
        topic_consistency_score = min(term_continuity / len(subsequent_messages), 1.0) if subsequent_messages else 1.0
        score += topic_consistency_score * 0.4
        checks['topic_consistency'] = topic_consistency_score > 0.3
    else:
        score += 0.4
        checks['topic_consistency'] = True
    
    # Check for proper turn-taking (user-assistant alternation)
    proper_alternation = True
    expected_role = 'user'
    for msg in conversation_history:
        if msg.get('role') not in ['user', 'assistant']:
            continue
        if msg.get('role') != expected_role:
            proper_alternation = False
            break
        expected_role = 'assistant' if expected_role == 'user' else 'user'
    
    if proper_alternation:
        score += 0.3
        checks['proper_turn_taking'] = True
    else:
        checks['proper_turn_taking'] = False
    
    # Check response length consistency
    assistant_responses = [msg for msg in conversation_history if msg.get('role') == 'assistant']
    if len(assistant_responses) > 1:
        lengths = [len(msg.get('content', '')) for msg in assistant_responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Lower variance is better (more consistent)
        consistency_score = 1.0 / (1.0 + (length_variance / (avg_length ** 2) if avg_length > 0 else 1))
        score += consistency_score * 0.3
        checks['response_length_consistency'] = consistency_score > 0.5
    else:
        score += 0.3
        checks['response_length_consistency'] = True
    
    return {
        'score': score,
        'name': 'conversation_coherence_check',
        'metadata': {
            'checks': checks,
            'turn_count': metadata.get('turn_count', 0),
            'reasoning': f'Passed {sum(checks.values())}/{len(checks)} coherence checks'
        }
    }

def context_retention_check(input, output, metadata):
    """
    Check if the system retains and uses context from previous turns.
    
    Args:
        input: The input conversation
        output: The generated response
        metadata: Contains conversation history
    
    Returns:
        dict: Score and metadata about context retention
    """
    conversation_history = metadata.get('conversation_history', [])
    
    if len(conversation_history) < 4:  # Need at least 2 full exchanges
        return {
            'score': 1.0,
            'name': 'context_retention_check',
            'metadata': {
                'reasoning': 'Not enough turns to evaluate context retention'
            }
        }
    
    score = 0
    checks = {}
    
    # Check if later responses reference earlier content
    early_messages = conversation_history[:len(conversation_history)//2]
    later_messages = conversation_history[len(conversation_history)//2:]
    
    # Extract meaningful content words from early messages
    early_content = ' '.join([msg.get('content', '') for msg in early_messages]).lower()
    early_words = set(early_content.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'how', 'when', 'where', 'why', 'who', 'i', 'you', 'it', 'this', 'that', 'these', 'those'}
    
    # Check how many early concepts appear in later messages
    later_content = ' '.join([msg.get('content', '') for msg in later_messages if msg.get('role') == 'assistant']).lower()
    
    if early_words and later_content:
        retained_words = sum(1 for word in early_words if word in later_content)
        retention_rate = retained_words / len(early_words)
        score += min(retention_rate * 2, 1.0) * 0.5  # Scale up since not all words need to be retained
        checks['references_earlier_content'] = retention_rate > 0.1
    else:
        checks['references_earlier_content'] = False
    
    # Check if responses build upon previous information
    assistant_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
    if len(assistant_messages) > 1:
        # Check for progressive elaboration (responses getting more specific)
        has_progression = False
        for i in range(1, len(assistant_messages)):
            prev_msg = assistant_messages[i-1].get('content', '')
            curr_msg = assistant_messages[i].get('content', '')
            
            # Simple check: does current message reference or expand on previous?
            if any(phrase in curr_msg.lower() for phrase in ['as mentioned', 'previously', 'earlier', 'before', 'already', 'addition', 'furthermore', 'also']):
                has_progression = True
                break
        
        if has_progression:
            score += 0.5
            checks['progressive_elaboration'] = True
        else:
            checks['progressive_elaboration'] = False
    
    return {
        'score': score,
        'name': 'context_retention_check',
        'metadata': {
            'checks': checks,
            'reasoning': f'System demonstrates {"good" if score > 0.5 else "limited"} context retention across turns'
        }
    }

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

# Run the evaluation
Eval(
    project_name,
    task=task,
    data=init_dataset(project=project_name, name="BraintrustMultiTurnQA"),
    scores=[
        document_retrieval_check,
        answer_relevance_check,
        answer_faithfulness_check,
        response_structure_check,
        conversation_coherence_check,
        context_retention_check
    ]
)