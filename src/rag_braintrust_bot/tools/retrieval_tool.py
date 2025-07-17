import braintrust
from pinecone import Pinecone
from pydantic import BaseModel
import voyageai
import os
from typing import List
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv('.env.local')

# Create a new Braintrust project
project = braintrust.projects.create("rag-braintrust-bot")

class Args(BaseModel):
    """
    Input arguments for the document retrieval tool.
    
    This class defines the expected input parameters for the retrieval function.
    Currently only requires a query string to search for relevant documents.
    
    Attributes:
        query (str): The search query to find relevant documents
    """
    query: str

class Document(BaseModel):
    """
    Represents a retrieved document section with its metadata.
    
    This class defines the structure of each document returned from Pinecone.
    It matches the structure used in the ingest.py script where documents
    are stored with a title and content.
    
    Attributes:
        title (str): The title/heading of the document section
        content (str): The actual content/text of the document section
        id (str): The unique identifier of the document
        score (float): The similarity score with the query
        tags (List[str]): Any tags associated with the document
    """
    title: str
    content: str
    id: str
    score: float
    tags: List[str] = []

class DocumentOutput(BaseModel):
    """
    Output structure for the document retrieval tool.
    
    This class defines the format of the response returned by the retrieval function.
    It wraps a list of Document objects in a 'documents' field for consistent API structure.
    
    Attributes:
        documents (List[Document]): List of retrieved document sections
    """
    documents: List[Document]

def handler(query: str):
    """
    Main handler function for document retrieval.
    
    This function:
    1. Initializes VoyageAI and Pinecone clients using API keys from environment
    2. Creates an embedding for the input query using VoyageAI
    3. Searches Pinecone for the most relevant documents
    4. Formats and returns the matching documents
    
    The function uses the same embedding model and index as specified in the ingest.py
    script to ensure consistency between storage and retrieval.
    
    Args:
        query (str): The search query to find relevant documents
        
    Returns:
        dict: A dictionary containing a list of matching documents with their titles and content
        
    Note:
        Requires VOYAGEAI_API_KEY and PINECONE_API_KEY to be set in environment variables
    """
    # Initialize VoyageAI client for embeddings
    vo = voyageai.Client(api_key=os.getenv("VOYAGEAI_API_KEY"))
    
    # Initialize Pinecone client and connect to the index
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("dev-rag-bot")
    
    # Use the same embedding model as specified in ingest.py
    MODEL = 'voyage-3'

    # Create embedding for the query
    xq = vo.embed(query, model=MODEL, input_type='query').embeddings[0]
    
    # Search Pinecone for similar documents
    response = index.query(
        vector=xq,
        top_k=3,
        include_metadata=True
    )
    
    # Format the matches to include title, content, and additional metadata
    matches = [{
        'title': match['metadata']['title'],
        'content': match['metadata']['content'],
        'id': match['id'],
        'score': match['score'],
        'tags': match['metadata'].get('tags', [])
    } for match in response['matches']]

    return {'documents': matches}

# Create the Braintrust tool for document retrieval
get_documents = project.tools.create(
    name="BT Get Documents", 
    handler=handler, 
    parameters=Args, 
    returns=DocumentOutput,
    slug='bt-get-documents-rag',
    if_exists='replace'
)
