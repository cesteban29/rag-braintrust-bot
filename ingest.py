import os
import re
import logging
import hashlib
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone, ServerlessSpec
import time
from datetime import datetime, timedelta

# Load environment variables from .env.local into process for API keys and settings
load_dotenv('.env.local')

# Constants for embedding model, Pinecone index name, and batch upload size
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'voyage-3')
INDEX_NAME = os.getenv('INDEX_NAME')  # No default value, must be set in .env.local
UPLOAD_BATCH_SIZE = int(os.getenv('UPLOAD_BATCH_SIZE', '50'))

# Configure logging to show more information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('ingest.log')  # Also save to file
    ]
)

@dataclass
class Section:
    # Represents a document section with a title and its corresponding content
    title: str
    content: str


def parse_markdown_file(file_path: str) -> List[Section]:
    """
    Parses a markdown/MDX file into sections based on headings.
    
    This function:
    - Opens and reads the file in UTF-8 encoding
    - Splits the content into lines
    - Iterates through each line looking for markdown headings (lines starting with 1-6 # symbols)
    - When it finds a heading, it:
        - Saves the previous section (if any) with its title and accumulated content
        - Starts a new section with the current heading as title
    - For non-heading lines, it adds them to the current section's content
    - Finally, it saves the last section and returns all non-empty sections
    - The function filters out any sections that have no content

    Args:
        file_path (str): Path to the markdown/MDX file to parse

    Returns:
        List[Section]: List of Section objects containing title and content pairs
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines()
    sections: List[Section] = []
    current_title = None
    current_content: List[str] = []
    for line in lines:
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            if current_title is not None:
                sections.append(Section(title=current_title, content='\n'.join(current_content).strip()))
            current_title = match.group(2)
            current_content = []
        else:
            if current_title is not None:
                current_content.append(line)
    if current_title is not None:
        sections.append(Section(title=current_title, content='\n'.join(current_content).strip()))
    return [s for s in sections if s.content]


def get_all_markdown_files(dir_path: str) -> List[str]:
    """
    Recursively finds all markdown files in a directory and its subdirectories.
    
    This function:
    - Takes a directory path as input
    - Uses os.walk() to traverse the directory tree
    - For each file found, checks if it ends with .md or .mdx
    - Returns a list of full paths to all markdown files found
    - This allows processing of markdown files in nested directories

    Args:
        dir_path (str): Root directory path to start searching from

    Returns:
        List[str]: List of full paths to all markdown files found
    """
    mdx_files: List[str] = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.md') or file.endswith('.mdx'):
                mdx_files.append(os.path.join(root, file))
    return mdx_files


def wait_for_index_ready(pc, index_name: str, timeout_seconds: int = 60) -> bool:
    """
    Wait for a Pinecone index to be ready with a timeout.
    
    Args:
        pc: Pinecone client
        index_name: Name of the index to check
        timeout_seconds: Maximum time to wait in seconds
        
    Returns:
        bool: True if index is ready, False if timeout occurred
    """
    start_time = datetime.now()
    timeout_time = start_time + timedelta(seconds=timeout_seconds)
    
    while datetime.now() < timeout_time:
        try:
            status = pc.describe_index(index_name)
            if status.status['ready']:
                return True
            logging.info(f"Index {index_name} not ready yet. Status: {status.status}")
            time.sleep(5)  # Wait 5 seconds between checks
        except Exception as e:
            logging.error(f"Error checking index status: {e}")
            return False
    
    logging.error(f"Timeout waiting for index {index_name} to be ready after {timeout_seconds} seconds")
    return False


def main():
    """
    Main orchestration function that processes markdown files and uploads embeddings to Pinecone.
    """
    start_time = time.time()
    logging.info("Starting ingestion process...")
    
    # Check if API keys are set
    if not os.getenv('VOYAGEAI_API_KEY'):
        logging.error("VOYAGEAI_API_KEY not found in environment variables")
        return
    if not os.getenv('PINECONE_API_KEY'):
        logging.error("PINECONE_API_KEY not found in environment variables")
        return

    # Initialize VoyageAI and Pinecone clients using API keys from environment
    logging.info("Initializing VoyageAI client...")
    try:
        vo = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
        logging.info("VoyageAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize VoyageAI client: {e}")
        return
    
    logging.info("Initializing Pinecone client...")
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        logging.info("Pinecone client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone client: {e}")
        return

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    logging.info(f"Looking for markdown files in: {data_dir}")
    
    try:
        mdx_files = get_all_markdown_files(data_dir)
        logging.info(f"Found {len(mdx_files)} markdown files: {[os.path.basename(f) for f in mdx_files]}")
    except Exception as e:
        logging.error(f"Failed to find markdown files: {e}")
        return

    all_sections: List[Section] = []
    for file_path in mdx_files:
        logging.info(f"Processing file: {file_path}")
        try:
            sections = parse_markdown_file(file_path)
            logging.info(f"Found {len(sections)} sections in {file_path}")
            all_sections.extend(sections)
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")
            continue

    if not all_sections:
        logging.error('No markdown sections found. Exiting.')
        return

    logging.info(f"Total sections found across all files: {len(all_sections)}")

    # Embed the first section to determine embedding dimension for index creation
    logging.info("Creating test embedding to determine dimension...")
    try:
        first = all_sections[0]
        logging.info(f"Creating test embedding for section: {first.title[:50]}...")
        first_embedding = vo.embed(f"# {first.title}\n{first.content}", model=EMBEDDING_MODEL, input_type='document').embeddings[0]
        dimension = len(first_embedding)
        logging.info(f"Test embedding created successfully. Dimension: {dimension}")
    except Exception as e:
        logging.error(f"Failed to create test embedding: {e}")
        return

    # Create the Pinecone index if it does not already exist
    try:
        logging.info(f"Creating Pinecone index: {INDEX_NAME}")
        try:
            # First check if index exists
            existing_index = pc.describe_index(INDEX_NAME)
            logging.info(f"Index {INDEX_NAME} already exists with status: {existing_index.status}")
            if not existing_index.status['ready']:
                logging.info("Waiting for existing index to be ready...")
                if not wait_for_index_ready(pc, INDEX_NAME):
                    logging.error("Existing index is not ready and timeout occurred")
                    return
        except Exception as e:
            # Index doesn't exist, create it
            logging.info(f"Index {INDEX_NAME} does not exist, creating it...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=dimension,
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            if not wait_for_index_ready(pc, INDEX_NAME):
                logging.error("New index creation timed out")
                return
            logging.info(f'Created and verified index {INDEX_NAME}')
    except Exception as e:
        logging.error(f"Failed to create or verify index {INDEX_NAME}: {e}")
        return

    try:
        index = pc.Index(INDEX_NAME)
        logging.info(f"Successfully connected to index {INDEX_NAME}")
    except Exception as e:
        logging.error(f"Failed to connect to index {INDEX_NAME}: {e}")
        return
    
    # Upsert embeddings in batches to Pinecone
    total_batches = (len(all_sections) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
    logging.info(f"Starting batch upload of {len(all_sections)} sections in {total_batches} batches")
    
    for i in range(0, len(all_sections), UPLOAD_BATCH_SIZE):
        batch_start_time = time.time()
        batch = all_sections[i:i+UPLOAD_BATCH_SIZE]
        vectors = []
        for sec in batch:
            try:
                text = f"# {sec.title}\n{sec.content}"
                logging.info(f"Creating embedding for section: {sec.title[:50]}...")
                embedding = vo.embed(text, model=EMBEDDING_MODEL, input_type='document').embeddings[0]
                vec_id = hashlib.md5((sec.title + text).encode()).hexdigest()
                vectors.append({
                    'id': vec_id,
                    'values': embedding,
                    'metadata': {'title': sec.title, 'content': sec.content}
                })
                logging.info(f"Successfully created embedding for section: {sec.title[:50]}")
            except Exception as e:
                logging.error(f"Failed to create embedding for section {sec.title[:50]}: {e}")
                continue

        if vectors:
            try:
                logging.info(f"Upserting batch {i//UPLOAD_BATCH_SIZE + 1}/{total_batches} with {len(vectors)} vectors")
                index.upsert(vectors)
                batch_duration = time.time() - batch_start_time
                logging.info(f'Successfully upserted batch {i//UPLOAD_BATCH_SIZE + 1}/{total_batches} in {batch_duration:.2f} seconds')
            except Exception as e:
                logging.error(f"Failed to upsert batch {i//UPLOAD_BATCH_SIZE + 1}: {e}")
                continue

    total_duration = time.time() - start_time
    logging.info(f'Successfully uploaded {len(all_sections)} sections to index {INDEX_NAME} in {total_duration:.2f} seconds')

if __name__ == '__main__':
    main()