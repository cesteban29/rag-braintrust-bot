import braintrust
import os
from pydantic import BaseModel

# Get project name from environment or use default
project_name = os.getenv('BRAINTRUST_PROJECT_NAME', 'rag-braintrust-bot')

# Create or get the project
project = braintrust.projects.create(name=project_name)

# Define the parameter schema for scorers
class ScorerParameters(BaseModel):
    output: str = ""
    expected: str = ""
    input: str = ""
    metadata: dict = {}


def safe_get_message_content(msg):
    """
    Safely extract content from a message, handling both dict and ChatCompletionMessage objects.
    """
    if hasattr(msg, 'content'):  # ChatCompletionMessage object
        return getattr(msg, 'content', '') or ''
    else:  # Dictionary format
        return msg.get('content', '') or ''


def safe_get_message_role(msg):
    """
    Safely extract role from a message, handling both dict and ChatCompletionMessage objects.
    """
    if hasattr(msg, 'role'):  # ChatCompletionMessage object
        return getattr(msg, 'role', '')
    else:  # Dictionary format
        return msg.get('role', '')

def rag_precision(input, output, expected, metadata):
    """
    Context-aware precision scorer handler.
    Measures what percentage of retrieved documents were actually used in the response.
    Handles both single-turn and multi-turn conversations.
    """
    if not metadata:
        return 0.0
        
    retrieved_docs = metadata.get('retrieved_documents', [])
    if not retrieved_docs:
        return 0.0
    
    # Handle multi-turn conversations
    if 'conversation_history' in metadata:
        # For multi-turn, check document usage across all assistant responses
        conversation_history = metadata.get('conversation_history', [])
        assistant_responses = [safe_get_message_content(msg) for msg in conversation_history if safe_get_message_role(msg) == 'assistant']
        
        if not assistant_responses:
            return 0.0
            
        # Combine all assistant responses for analysis
        combined_output = ' '.join(assistant_responses).lower()
    else:
        # Single-turn: use the direct output
        combined_output = output.lower() if output else ''
    
    if not combined_output:
        return 0.0
    
    # Check which documents were actually referenced/used
    used_docs = 0
    
    for doc in retrieved_docs:
        doc_content = doc.get('content', '').lower()
        doc_title = doc.get('title', '').lower()
        
        # Extract key phrases from document
        doc_words = set(word for word in doc_content.split() if len(word) > 4)
        output_words = set(word for word in combined_output.split() if len(word) > 4)
        overlap = doc_words.intersection(output_words)
        
        # Consider document used if significant overlap
        overlap_ratio = len(overlap) / len(doc_words) if doc_words else 0
        if overlap_ratio > 0.1 or doc_title in combined_output:
            used_docs += 1
    
    return used_docs / len(retrieved_docs)


def rag_recall(input, output, expected, metadata):
    """
    Context-aware recall scorer handler.
    Measures what percentage of relevant information was included in the response.
    Handles both single-turn and multi-turn conversations.
    """
    if not metadata:
        metadata = {}
    
    # Extract query - handle both single-turn and multi-turn contexts
    if 'conversation_history' in metadata:
        # For multi-turn, extract the original query from conversation history
        conversation_history = metadata.get('conversation_history', [])
        user_messages = [safe_get_message_content(msg) for msg in conversation_history if safe_get_message_role(msg) == 'user']
        
        if user_messages:
            # Use the first user message as the main query, combine with later questions
            query = ' '.join(user_messages)
        else:
            query = str(input) if input else ''
            
        # Combine all assistant responses for analysis
        assistant_responses = [safe_get_message_content(msg) for msg in conversation_history if safe_get_message_role(msg) == 'assistant']
        combined_output = ' '.join(assistant_responses) if assistant_responses else (output or '')
    else:
        # Single-turn: extract query from input
        if isinstance(input, dict):
            query = input.get('query', input.get('input', ''))
        else:
            query = str(input) if input else ''
        combined_output = output or ''
    
    if not combined_output or not query:
        return 0.0
    
    query_lower = query.lower()
    stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'does', 'is', 'are', 'the', 'a', 'an'}
    
    # Extract key concepts from all user queries
    query_words = query_lower.split()
    key_concepts = [w for w in query_words if len(w) > 3 and w not in stop_words]
    
    if not key_concepts:
        return 1.0
    
    # Check which concepts are covered in the combined output
    output_lower = combined_output.lower()
    covered_concepts = sum(1 for concept in key_concepts 
                          if concept in output_lower or concept[:-1] in output_lower)
    
    return covered_concepts / len(key_concepts)


def rag_f1(input, output, expected, metadata):
    """
    Context-aware F1 scorer handler.
    Harmonic mean of precision and recall.
    Works with both single-turn and multi-turn conversations.
    """
    precision = rag_precision(input, output, expected, metadata)
    recall = rag_recall(input, output, expected, metadata)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# RAG Precision Scorer
project.scorers.create(
    name="RAG Precision",
    slug="rag-precision", 
    description="Context-aware precision: percentage of retrieved documents actually used",
    parameters=ScorerParameters,
    handler=rag_precision,
)

# RAG Recall Scorer  
project.scorers.create(
    name="RAG Recall",
    slug="rag-recall",
    description="Context-aware recall: percentage of relevant information included",
    parameters=ScorerParameters,
    handler=rag_recall,
)

# RAG F1 Scorer
project.scorers.create(
    name="RAG F1",
    slug="rag-f1", 
    description="Context-aware F1: harmonic mean of precision and recall",
    parameters=ScorerParameters,
    handler=rag_f1,
)

# RAG Factuality LLM Scorer
project.scorers.create(
    name="RAG Factuality LLM",
    slug="rag-factuality-llm",
    description="Checks if the output is factually consistent with retrieved documents",
    messages=[
        {
            "role": "system",
            "content": "You are a factuality checker. You will be given a response and the source documents it should be based on."
        },
        {
            "role": "user",
            "content": """Given the following retrieved documents:
{{metadata.retrieved_documents}}

And this response:
{{output}}

Is the response factually consistent with the information in the documents? 
- Return "A" if the response is fully factual and supported by the documents
- Return "B" if the response contains some factual information but also unsupported claims
- Return "C" if the response is mostly or entirely unsupported by the documents

Think step by step about which facts in the response are supported by the documents.""",
        },
    ],
    model="gpt-4o-mini",
    use_cot=True,
    choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
)

# RAG Relevance LLM Scorer
project.scorers.create(
    name="RAG Relevance LLM",
    slug="rag-relevance-llm",
    description="Checks if the output is relevant to the input query",
    messages=[
        {
            "role": "user",
            "content": """Query: {{input}}

Response: {{output}}

Does the response relevantly address the query?
- Return "A" if the response directly and fully addresses the query
- Return "B" if the response partially addresses the query or includes some irrelevant information
- Return "C" if the response does not address the query or is mostly irrelevant

Think step by step about how well the response answers the specific question asked.""",
        },
    ],
    model="gpt-4o-mini",
    use_cot=True,
    choice_scores={"A": 1.0, "B": 0.5, "C": 0.0},
)

# RAG Completeness LLM Scorer
project.scorers.create(
    name="RAG Completeness LLM", 
    slug="rag-completeness-llm",
    description="Evaluates if the response completely addresses all aspects of the query",
    messages=[
        {
            "role": "user",
            "content": """Query: {{input}}

Response: {{output}}

Does the response completely address all aspects of the query?
- Return "A" if all parts of the query are thoroughly addressed
- Return "B" if most parts are addressed but some aspects are missing or incomplete
- Return "C" if significant parts of the query are not addressed

Consider:
1. Are all questions answered?
2. Are all requested items provided?
3. Is the response thorough without being excessive?""",
        },
    ],
    model="gpt-4o-mini",
    use_cot=True,
    choice_scores={"A": 1.0, "B": 0.6, "C": 0.2},
)