import braintrust
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv('.env.local')

# Get project name from environment or use default
project_name = os.getenv('BRAINTRUST_PROJECT_NAME', 'rag-braintrust-bot')

# Create or get the project
project = braintrust.projects.create(name=project_name)

# Define the parameter schema for scorers
class ScorerParameters(BaseModel):
    output: str = ""
    expected: str = ""
    input: str = ""  # Match the actual log field name
    metadata: dict = {}

def document_retrieval_check(output, expected, input, metadata):
    """
    Check if the correct documents were retrieved for the query.
    
    Args:
        output: The generated response
        expected: Expected response from dataset
        input: The input query
        metadata: Contains tool call information
    
    Returns:
        dict: Score and metadata about document retrieval accuracy
    """
    # Handle None values
    if output is None:
        output = ""
    if metadata is None:
        metadata = {}
    
    # Extract retrieved document information from metadata
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

def answer_relevance_check(output, expected, input, metadata):
    """
    Check if the generated answer is relevant to the input query.
    Uses a simple heuristic based on keyword overlap and length.
    
    Args:
        output: The generated response
        expected: Expected response from dataset
        input: The input query
        metadata: Additional metadata
    
    Returns:
        dict: Score and metadata about answer relevance
    """
    # Handle None values first
    if output is None:
        output = ""
    if metadata is None:
        metadata = {}
    
    # Handle both string and dictionary input formats, and None values
    if input is None:
        query = ""
    elif isinstance(input, str):
        query = input
    elif isinstance(input, dict):
        query = input.get('query', input.get('input', ''))
    else:
        query = str(input) if input is not None else ""
    
    if not output or not query:
        return {
            'score': 0,
            'name': 'answer_relevance_check',
            'metadata': {
                'reasoning': 'Empty output or query'
            }
        }
    
    # Simple keyword overlap scoring
    query_words = set(query.lower().split())
    output_words = set(output.lower().split())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    query_words = query_words - stopwords
    output_words = output_words - stopwords
    
    if not query_words:
        overlap_score = 0.5  # Neutral score if no meaningful query words
    else:
        overlap = len(query_words.intersection(output_words))
        overlap_score = overlap / len(query_words)
    
    # Length check - penalize very short responses
    length_score = min(len(output) / 100, 1.0)  # Normalize to 1 for 100+ chars
    
    # Combine scores
    final_score = (overlap_score * 0.7) + (length_score * 0.3)
    
    return {
        'score': final_score,
        'name': 'answer_relevance_check',
        'metadata': {
            'query_words': list(query_words),
            'overlap_count': len(query_words.intersection(output_words)),
            'overlap_score': overlap_score,
            'length_score': length_score,
            'output_length': len(output)
        }
    }

def answer_faithfulness_check(output, expected, input, metadata):
    """
    Check if the answer is faithful to the retrieved documents.
    
    Args:
        output: The generated response
        expected: Expected response from dataset
        input: The input query
        metadata: Contains retrieved document information
    
    Returns:
        dict: Score and metadata about answer faithfulness
    """
    # Handle None values
    if output is None:
        output = ""
    if metadata is None:
        metadata = {}
    # Extract document content from metadata
    document_contents = []
    retrieved_docs = metadata.get('retrieved_documents', [])
    
    for doc in retrieved_docs:
        if 'content' in doc:
            document_contents.append(doc['content'])
    
    if not document_contents:
        return {
            'score': 0.5,  # Neutral score if no context available
            'name': 'answer_faithfulness_check',
            'metadata': {
                'reasoning': 'No retrieved documents available for faithfulness check'
            }
        }
    
    # Combine all document content
    context = ' '.join(document_contents)
    
    # Simple faithfulness check based on content overlap
    context_words = set(context.lower().split())
    output_words = set(output.lower().split())
    
    # Remove stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    context_words = context_words - stopwords
    output_words = output_words - stopwords
    
    if not output_words:
        return {
            'score': 0,
            'name': 'answer_faithfulness_check',
            'metadata': {
                'reasoning': 'Empty output'
            }
        }
    
    # Calculate what percentage of output words are supported by context
    supported_words = output_words.intersection(context_words)
    faithfulness_score = len(supported_words) / len(output_words)
    
    return {
        'score': faithfulness_score,
        'name': 'answer_faithfulness_check',
        'metadata': {
            'total_output_words': len(output_words),
            'supported_words': len(supported_words),
            'faithfulness_score': faithfulness_score,
            'unsupported_words': list(output_words - context_words)[:10]  # First 10 unsupported words
        }
    }

def response_structure_check(output, expected, input, metadata):
    """
    Check if the response follows good documentation answer structure.
    
    Args:
        output: The generated response
        expected: Expected response from dataset
        input: The input query
        metadata: Additional metadata
    
    Returns:
        dict: Score and metadata about response structure
    """
    # Handle None values
    if output is None:
        output = ""
    if not output:
        return {
            'score': 0,
            'name': 'response_structure_check',
            'metadata': {
                'reasoning': 'Empty output'
            }
        }
    
    score = 0
    checks = {}
    
    # Check for adequate length (not too short)
    if len(output) > 50:
        score += 0.2
        checks['adequate_length'] = True
    else:
        checks['adequate_length'] = False
    
    # Check for structured formatting (bullet points, numbers, etc.)
    if any(marker in output for marker in ['•', '-', '*', '1.', '2.', '3.']):
        score += 0.2
        checks['has_formatting'] = True
    else:
        checks['has_formatting'] = False
    
    # Check for code examples (backticks)
    if '`' in output:
        score += 0.2
        checks['has_code_examples'] = True
    else:
        checks['has_code_examples'] = False
    
    # Check for clear sentence structure (periods)
    if output.count('.') >= 2:
        score += 0.2
        checks['clear_sentences'] = True
    else:
        checks['clear_sentences'] = False
    
    # Check for helpful details (mentions specific features/tools)
    braintrust_terms = ['braintrust', 'evaluation', 'dataset', 'prompt', 'logging', 'tracing', 'playground']
    if any(term in output.lower() for term in braintrust_terms):
        score += 0.2
        checks['mentions_braintrust_features'] = True
    else:
        checks['mentions_braintrust_features'] = False
    
    return {
        'score': score,
        'name': 'response_structure_check',
        'metadata': {
            'checks': checks,
            'output_length': len(output),
            'reasoning': f'Passed {sum(checks.values())}/5 structure checks'
        }
    }

# Create all the scorers
print(f"Pushing scorers to Braintrust project: {project_name}")

# Document Retrieval Check Scorer
print("Creating document_retrieval_check scorer...")
project.scorers.create(
    name="Document Retrieval Check",
    slug="document-retrieval-check",
    description="Evaluates the quality and relevance of retrieved documents for a given query",
    parameters=ScorerParameters,
    handler=document_retrieval_check,
)

# Answer Relevance Check Scorer
print("Creating answer_relevance_check scorer...")
project.scorers.create(
    name="Answer Relevance Check",
    slug="answer-relevance-check", 
    description="Checks if the generated answer is relevant to the input query using keyword overlap and length heuristics",
    parameters=ScorerParameters,
    handler=answer_relevance_check,
)

# Answer Faithfulness Check Scorer
print("Creating answer_faithfulness_check scorer...")
project.scorers.create(
    name="Answer Faithfulness Check",
    slug="answer-faithfulness-check",
    description="Verifies that the answer is faithful to the retrieved documents by checking content overlap",
    parameters=ScorerParameters,
    handler=answer_faithfulness_check,
)

# Response Structure Check Scorer
print("Creating response_structure_check scorer...")
project.scorers.create(
    name="Response Structure Check", 
    slug="response-structure-check",
    description="Evaluates if the response follows good documentation answer structure (formatting, code examples, etc.)",
    parameters=ScorerParameters,
    handler=response_structure_check,
)

print("✅ All scorers have been successfully pushed to Braintrust!")
print(f"You can view them in your Braintrust project: {project_name}") 