import os
import re
import logging
import hashlib
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone
import time

# Load environment variables from .env.local
load_dotenv('.env.local')

# Constants
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'voyage-3')
INDEX_NAME = os.getenv('INDEX_NAME')
UPLOAD_BATCH_SIZE = int(os.getenv('UPLOAD_BATCH_SIZE', '50'))
BRAINTRUST_DOCS_URL = "https://www.braintrust.dev/docs/llms.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingest_web.log')
    ]
)

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation with metadata"""
    id: str
    title: str
    content: str
    source: str
    url: str
    date: Optional[str] = None
    section_type: Optional[str] = None  # 'changelog', 'docs', 'api', etc.


def fetch_document(url: str) -> str:
    """Fetch document content from URL"""
    try:
        logging.info(f"Fetching document from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logging.info(f"Successfully fetched {len(response.text)} characters")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch document: {e}")
        raise


def parse_braintrust_docs(content: str) -> List[DocumentChunk]:
    """
    Parse Braintrust documentation into chunks.
    Strategy: Split by date headers for changelogs, and by sections for docs.
    """
    chunks: List[DocumentChunk] = []
    lines = content.splitlines()
    
    current_date = None
    current_title = None
    current_content = []
    current_section_type = "changelog"
    
    # Patterns for different section types
    date_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})$')
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)')
    
    for i, line in enumerate(lines):
        # Check if this is a date line (changelog entry)
        date_match = date_pattern.match(line.strip())
        if date_match:
            # Save previous chunk if exists
            if current_title and current_content:
                chunk_content = '\n'.join(current_content).strip()
                if chunk_content:
                    chunk_id = hashlib.md5(f"{current_title}{chunk_content}".encode()).hexdigest()
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        title=current_title,
                        content=chunk_content,
                        source="braintrust_docs",
                        url=BRAINTRUST_DOCS_URL,
                        date=current_date,
                        section_type=current_section_type
                    ))
            
            # Start new changelog chunk
            current_date = date_match.group(1)
            current_title = f"Braintrust Update - {current_date}"
            current_content = []
            current_section_type = "changelog"
            continue
        
        # Check if this is a header (for documentation sections)
        header_match = header_pattern.match(line)
        if header_match and not current_date:  # Only treat as header if not in a dated section
            # Save previous chunk if exists
            if current_title and current_content:
                chunk_content = '\n'.join(current_content).strip()
                if chunk_content:
                    chunk_id = hashlib.md5(f"{current_title}{chunk_content}".encode()).hexdigest()
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        title=current_title,
                        content=chunk_content,
                        source="braintrust_docs",
                        url=BRAINTRUST_DOCS_URL,
                        date=current_date,
                        section_type=current_section_type
                    ))
            
            # Start new documentation chunk
            current_title = header_match.group(2)
            current_content = []
            current_section_type = detect_section_type(current_title)
            continue
        
        # Add line to current content
        if current_title:
            current_content.append(line)
            
            # Create chunk if content is getting too long (>1500 chars)
            if len('\n'.join(current_content)) > 1500:
                chunk_content = '\n'.join(current_content).strip()
                if chunk_content:
                    chunk_id = hashlib.md5(f"{current_title}{chunk_content}".encode()).hexdigest()
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        title=current_title,
                        content=chunk_content,
                        source="braintrust_docs",
                        url=BRAINTRUST_DOCS_URL,
                        date=current_date,
                        section_type=current_section_type
                    ))
                    # Continue with same title but reset content
                    current_content = []
    
    # Save final chunk
    if current_title and current_content:
        chunk_content = '\n'.join(current_content).strip()
        if chunk_content:
            chunk_id = hashlib.md5(f"{current_title}{chunk_content}".encode()).hexdigest()
            chunks.append(DocumentChunk(
                id=chunk_id,
                title=current_title,
                content=chunk_content,
                source="braintrust_docs",
                url=BRAINTRUST_DOCS_URL,
                date=current_date,
                section_type=current_section_type
            ))
    
    logging.info(f"Parsed {len(chunks)} chunks from document")
    return chunks


def detect_section_type(title: str) -> str:
    """Detect the type of section based on title"""
    title_lower = title.lower()
    if any(word in title_lower for word in ['api', 'endpoint', 'rest']):
        return 'api'
    elif any(word in title_lower for word in ['sdk', 'python', 'typescript', 'node']):
        return 'sdk'
    elif any(word in title_lower for word in ['trace', 'tracing', 'log', 'logging']):
        return 'tracing'
    elif any(word in title_lower for word in ['eval', 'evaluation', 'experiment']):
        return 'evaluation'
    elif any(word in title_lower for word in ['prompt', 'playground']):
        return 'prompts'
    elif any(word in title_lower for word in ['proxy', 'ai proxy']):
        return 'proxy'
    else:
        return 'docs'


def should_update_chunk(index, chunk: DocumentChunk, existing_ids: set) -> bool:
    """
    Determine if a chunk should be updated in the index.
    For now, we'll use a simple strategy: update if the ID is different.
    """
    return chunk.id not in existing_ids


def get_existing_ids(index) -> set:
    """Get all existing vector IDs from the index"""
    try:
        # Query to get stats about the index
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors == 0:
            return set()
        
        logging.info(f"Index contains {total_vectors} vectors")
        
        # For now, we'll do a simple approach: 
        # If we're re-ingesting, we'll update everything
        # In production, you might want to implement a more sophisticated check
        return set()
        
    except Exception as e:
        logging.warning(f"Could not retrieve existing IDs: {e}")
        return set()


def main(clear_index: bool = False):
    """
    Main function to ingest Braintrust documentation from web.
    This is the primary ingestion method - replaces local file ingestion.
    
    Args:
        clear_index: If True, clear the entire index before ingesting
    """
    start_time = time.time()
    logging.info("Starting Braintrust documentation ingestion from web...")
    
    # Check API keys
    if not os.getenv('VOYAGEAI_API_KEY'):
        logging.error("VOYAGEAI_API_KEY not found")
        return
    if not os.getenv('PINECONE_API_KEY'):
        logging.error("PINECONE_API_KEY not found")
        return
    
    # Initialize clients
    try:
        vo = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        logging.info("Clients initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize clients: {e}")
        return
    
    # Fetch and parse document
    try:
        content = fetch_document(BRAINTRUST_DOCS_URL)
        chunks = parse_braintrust_docs(content)
        logging.info(f"Parsed {len(chunks)} chunks from document")
        
        # Log section type distribution
        section_types = {}
        for chunk in chunks:
            section_types[chunk.section_type] = section_types.get(chunk.section_type, 0) + 1
        logging.info(f"Section types: {section_types}")
        
    except Exception as e:
        logging.error(f"Failed to fetch or parse document: {e}")
        return
    
    if not chunks:
        logging.error("No chunks to process")
        return
    
    # Connect to index
    try:
        index = pc.Index(INDEX_NAME)
        logging.info(f"Connected to index {INDEX_NAME}")
        
        # Clear index if requested
        if clear_index:
            logging.warning("Clearing entire index as requested...")
            try:
                # Delete all vectors in the index
                index.delete(delete_all=True)
                logging.info("Index cleared successfully")
                existing_ids = set()
            except Exception as e:
                logging.error(f"Failed to clear index: {e}")
                return
        else:
            # Get existing IDs to check for updates
            existing_ids = get_existing_ids(index)
        
    except Exception as e:
        logging.error(f"Failed to connect to index: {e}")
        return
    
    # Create embeddings and upsert in batches
    vectors_to_upsert = []
    chunks_processed = 0
    chunks_skipped = 0
    
    for i in range(0, len(chunks), UPLOAD_BATCH_SIZE):
        batch = chunks[i:i+UPLOAD_BATCH_SIZE]
        batch_vectors = []
        
        for chunk in batch:
            # Skip if chunk already exists and hasn't changed
            if not should_update_chunk(index, chunk, existing_ids):
                chunks_skipped += 1
                logging.debug(f"Skipping unchanged chunk: {chunk.title}")
                continue
            
            try:
                # Create text for embedding
                text = f"# {chunk.title}\n{chunk.content}"
                
                # Create embedding
                embedding = vo.embed(text, model=EMBEDDING_MODEL, input_type='document').embeddings[0]
                
                # Prepare vector with metadata
                metadata = {
                    'title': chunk.title[:500],  # Pinecone has metadata size limits
                    'content': chunk.content[:2000],  # Truncate if needed
                    'source': chunk.source,
                    'url': chunk.url,
                    'section_type': chunk.section_type
                }
                
                if chunk.date:
                    metadata['date'] = chunk.date
                
                batch_vectors.append({
                    'id': chunk.id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                chunks_processed += 1
                
            except Exception as e:
                logging.error(f"Failed to process chunk '{chunk.title}': {e}")
                continue
        
        # Upsert batch if we have vectors
        if batch_vectors:
            try:
                index.upsert(batch_vectors)
                logging.info(f"Upserted batch of {len(batch_vectors)} vectors")
                vectors_to_upsert.extend(batch_vectors)
            except Exception as e:
                logging.error(f"Failed to upsert batch: {e}")
    
    # Log summary
    total_time = time.time() - start_time
    logging.info(f"""
    Ingestion complete:
    - Total chunks: {len(chunks)}
    - Chunks processed: {chunks_processed}
    - Chunks skipped: {chunks_skipped}
    - Vectors upserted: {len(vectors_to_upsert)}
    - Time taken: {total_time:.2f} seconds
    """)
    
    print(f"✅ Successfully ingested {chunks_processed} chunks from Braintrust docs")
    print(f"📊 Skipped {chunks_skipped} unchanged chunks")
    print(f"⏱️  Completed in {total_time:.2f} seconds")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ingest Braintrust documentation from web')
    parser.add_argument('--clear', action='store_true', help='Clear entire index before ingesting (removes ALL documents)')
    args = parser.parse_args()
    
    main(clear_index=args.clear)