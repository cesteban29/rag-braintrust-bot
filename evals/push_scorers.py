import braintrust
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

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

def precision_handler(input, output, expected, metadata):
    """
    Precision scorer handler.
    Measures what percentage of retrieved documents were actually used in the response.
    """
    retrieved_docs = metadata.get('retrieved_documents', []) if metadata else []
    
    if not retrieved_docs:
        return 0.0
    
    # Check which documents were actually referenced/used in the output
    used_docs = 0
    output_lower = output.lower() if output else ''
    
    for doc in retrieved_docs:
        doc_content = doc.get('content', '').lower()
        doc_title = doc.get('title', '').lower()
        
        # Extract key phrases from document
        doc_words = set(word for word in doc_content.split() if len(word) > 4)
        output_words = set(word for word in output_lower.split() if len(word) > 4)
        overlap = doc_words.intersection(output_words)
        
        # Consider document used if significant overlap
        overlap_ratio = len(overlap) / len(doc_words) if doc_words else 0
        if overlap_ratio > 0.1 or doc_title in output_lower:
            used_docs += 1
    
    return used_docs / len(retrieved_docs) if retrieved_docs else 0.0


def recall_handler(input, output, expected, metadata):
    """
    Recall scorer handler.
    Measures what percentage of relevant information was included in the response.
    """
    # Extract query from input
    if isinstance(input, dict):
        query = input.get('query', input.get('input', ''))
    else:
        query = str(input)
    
    if not output or not query:
        return 0.0
    
    query_lower = query.lower()
    stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'does', 'is', 'are', 'the', 'a', 'an'}
    
    # Extract key concepts
    query_words = query_lower.split()
    key_concepts = [w for w in query_words if len(w) > 3 and w not in stop_words]
    
    if not key_concepts:
        return 1.0
    
    # Check which concepts are covered
    output_lower = output.lower()
    covered_concepts = sum(1 for concept in key_concepts 
                          if concept in output_lower or concept[:-1] in output_lower)
    
    return covered_concepts / len(key_concepts)


def f1_handler(input, output, expected, metadata):
    """
    F1 scorer handler.
    Harmonic mean of precision and recall.
    """
    precision = precision_handler(input, output, expected, metadata)
    recall = recall_handler(input, output, expected, metadata)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# Create all the scorers
print(f"Pushing scorers to Braintrust project: {project_name}")

# RAG Precision Scorer
print("Creating precision_handler scorer...")
project.scorers.create(
    name="RAG Precision",
    slug="rag-precision", 
    description="Measures what percentage of retrieved documents were actually used in the response",
    parameters=ScorerParameters,
    handler=precision_handler,
)

# RAG Recall Scorer  
print("Creating recall_handler scorer...")
project.scorers.create(
    name="RAG Recall",
    slug="rag-recall",
    description="Measures what percentage of relevant information was included in the response",
    parameters=ScorerParameters,
    handler=recall_handler,
)

# RAG F1 Scorer
print("Creating f1_handler scorer...")
project.scorers.create(
    name="RAG F1",
    slug="rag-f1", 
    description="Harmonic mean of precision and recall for balanced evaluation",
    parameters=ScorerParameters,
    handler=f1_handler,
)

# RAG Factuality LLM Scorer
print("Creating RAG Factuality LLM scorer...")
project.scorers.create(
    name="RAG Factuality",
    slug="rag-factuality",
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
print("Creating RAG Relevance LLM scorer...")
project.scorers.create(
    name="RAG Relevance",
    slug="rag-relevance",
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
print("Creating RAG Completeness LLM scorer...")
project.scorers.create(
    name="RAG Completeness", 
    slug="rag-completeness",
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

print("✅ All scorers have been successfully pushed to Braintrust!")
print(f"You can view them in your Braintrust project: {project_name}")