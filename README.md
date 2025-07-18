# RAG Braintrust Bot

A Retrieval-Augmented Generation (RAG) system built with Braintrust for evaluation, using VoyageAI for embeddings and Pinecone for vector storage.

## Overview

This project creates a RAG system that:
- Processes markdown documents and stores them as vector embeddings
- Provides semantic search capabilities over the document collection
- Uses Braintrust for comprehensive evaluation and monitoring
- Includes automated evaluation with multiple quality metrics
- Supports interactive querying and testing

## Prerequisites

- Python 3.12 or higher
- API keys for VoyageAI, Pinecone, and Braintrust
- Homebrew (for macOS setup)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-braintrust-bot
```

### 2. Environment Setup

For macOS users, run the automated setup script:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

For manual setup:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env.local` file in the project root with your API keys:

```bash
# VoyageAI API Key
VOYAGEAI_API_KEY=your_voyage_ai_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_key_here

# Braintrust API Key
BRAINTRUST_API_KEY=your_braintrust_key_here

# Project Configuration
BRAINTRUST_PROJECT_NAME=rag-braintrust-bot
INDEX_NAME=dev-rag-bot
EMBEDDING_MODEL=voyage-3
UPLOAD_BATCH_SIZE=50
```

### 4. Set up Pinecone Index

Create an index in your Pinecone console with the following specifications:
- Index name: `dev-rag-bot` (or match your `INDEX_NAME` in `.env.local`)
- Dimension: 1024 (for voyage-3 model)
- Metric: cosine
- Pod type: Serverless (recommended for development)

## Usage

### 1. Ingest Documents

Place your markdown files in the `data/documents/` directory, then run:

```bash
source venv/bin/activate
python src/rag_braintrust_bot/ingestion/ingest.py
```

This will:
- Parse all markdown files in the documents directory
- Create embeddings using VoyageAI
- Store the embeddings in Pinecone
- Log the process to `ingest.log`

### 2. Create Braintrust Tools

Set up the document retrieval tool:

```bash
python src/rag_braintrust_bot/retrieval/retrieval_tool.py
```

#### 3. Push Scorers to Braintrust

```bash
braintrust push src/rag_braintrust_bot/braintrust_resources/push_scorers.py --if-exists replace
```

### 4. Run RAG System Demo - these will appear in Braintrust Logs

Test the RAG system with sample queries:

```bash
python src/rag_braintrust_bot/rag_demo.py
```

### 5. Run Offline Evaluations

Evaluate the RAG system performance using Braintrust:

```bash
# Load environment variables and run evaluation
export $(grep -v '^#' .env.local | xargs) && braintrust eval evals/eval_test.py
```

This will:
- Process 30 test questions about Braintrust documentation
- Evaluate document retrieval quality
- Assess answer relevance and faithfulness
- Check response structure and formatting
- Generate detailed performance metrics

## Project Structure

```
rag-braintrust-bot/
├── src/
│   └── rag_braintrust_bot/           # Main package
│       ├── ingestion/
│       │   └── ingest.py             # Document ingestion script
│       ├── tools/
│       │   └── retrieval_tool.py     # Document retrieval tool
│       ├── braintrust/
│       │   └── rag_evaluation.py     # Multi-turn conversation evaluation
│       └── rag_demo.py               # Main demo script
├── data/
│   └── documents/                    # Markdown files to be processed
├── evals/                            # Braintrust evaluation files
│   ├── datasets/
│   │   └── braintrust_documentation_qa.json  # Test questions dataset
│   └── eval_test.py              # Main evaluation script
├── scripts/
│   └── setup.sh                      # Automated setup script
├── venv/                             # Virtual environment
├── .env.local                        # Environment variables (create this)
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Python dependencies
└── LICENSE                           # MIT License
```

## Key Components

### Document Ingestion
- **File**: `src/rag_braintrust_bot/ingestion/ingest.py`
- **Purpose**: Processes markdown files and creates vector embeddings
- **Features**: Automatic section parsing, batch processing, error handling

### Document Retrieval
- **File**: `src/rag_braintrust_bot/retrieval/retrieval_tool.py`
- **Purpose**: Provides semantic search over the document collection
- **Features**: Returns relevant documents with similarity scores

### RAG System Demo
- **File**: `src/rag_braintrust_bot/rag_demo.py`
- **Purpose**: Interactive demonstration of the RAG system
- **Features**: Processes sample queries, retrieves documents, generates answers

### Evaluation System
- **File**: `evals/eval_test.py`
- **Purpose**: Comprehensive evaluation of RAG system performance
- **Features**: Multi-metric evaluation (retrieval, relevance, faithfulness, structure)
- **Dataset**: 30 curated questions about Braintrust documentation

#### Evaluation Metrics

The evaluation system measures four key aspects:

1. **Document Retrieval Check**: Evaluates whether relevant documents are retrieved for each query
2. **Answer Relevance Check**: Measures how well the generated answer addresses the input question
3. **Answer Faithfulness Check**: Assesses whether the answer is grounded in the retrieved documents
4. **Response Structure Check**: Validates proper formatting, code examples, and documentation structure

## Dependencies

### Core Libraries
- `braintrust`: Evaluation and prompt management
- `voyageai`: Text embeddings
- `pinecone`: Vector database
- `beautifulsoup4`: HTML/XML parsing
- `autoevals`: Evaluation metrics

### Development Tools
- `python-dotenv`: Environment variable management
- `openai`: OpenAI API integration
- `pydantic`: Data validation

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in `.env.local`
2. **Pinecone Index Not Found**: Create the index in Pinecone console before running ingestion
3. **Python Version**: Ensure you're using Python 3.12 or higher
4. **Virtual Environment**: Always activate the virtual environment before running scripts
5. **Evaluation Command**: Use `export $(grep -v '^#' .env.local | xargs) && braintrust eval evals/eval_test.py` to ensure environment variables are loaded
6. **OpenAI Compatibility**: If you encounter OpenAI version issues, the evaluation system uses the latest compatible versions

### Document Ingest Logging

- Ingestion logs are saved to `ingest.log`
- Use `logging.info()` level for detailed processing information
- Check console output for real-time status updates

## Contributing

1. Ensure all tests pass before submitting changes
2. Follow the existing code style and structure
3. Update documentation for any new features
4. Add appropriate error handling and logging

## License

This project is licensed under the MIT License.

