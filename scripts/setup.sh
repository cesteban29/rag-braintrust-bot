#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up RAG Braintrust Bot development environment..."

# Check if we're on macOS and detect Python version
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check for Python 3.12+ (preferred) or 3.11+
    PYTHON_CMD=""
    
    # Try Python 3.12 first (recommended)
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        echo "✅ Found Python 3.12"
    elif [ -f "/opt/homebrew/opt/python@3.12/bin/python3.12" ]; then
        PYTHON_CMD="/opt/homebrew/opt/python@3.12/bin/python3.12"
        echo "✅ Found Python 3.12 via Homebrew"
    elif [ -f "/usr/local/opt/python@3.12/bin/python3.12" ]; then
        PYTHON_CMD="/usr/local/opt/python@3.12/bin/python3.12"
        echo "✅ Found Python 3.12 via Homebrew (Intel Mac)"
    # Fall back to Python 3.11
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo "✅ Found Python 3.11"
    elif [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
        PYTHON_CMD="/opt/homebrew/opt/python@3.11/bin/python3.11"
        echo "✅ Found Python 3.11 via Homebrew"
    elif [ -f "/usr/local/opt/python@3.11/bin/python3.11" ]; then
        PYTHON_CMD="/usr/local/opt/python@3.11/bin/python3.11"
        echo "✅ Found Python 3.11 via Homebrew (Intel Mac)"
    # Try generic python3
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
        if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
            PYTHON_CMD="python3"
            echo "✅ Found Python $PYTHON_VERSION"
        else
            echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.11 or higher."
            echo "   Install with: brew install python@3.12"
            exit 1
        fi
    else
        echo "❌ Python 3.11+ not found. Please install it with:"
        echo "   brew install python@3.12"
        echo "   or"
        echo "   brew install python@3.11"
        exit 1
    fi
else
    # Non-macOS systems
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "✅ Found Python 3"
    else
        echo "❌ Python 3 not found. Please install Python 3.11 or higher."
        exit 1
    fi
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📚 Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Create .vscode directory if it doesn't exist
if [ ! -d ".vscode" ]; then
    echo "📁 Creating .vscode directory..."
    mkdir .vscode
fi

# Create settings.json for VS Code
echo "⚙️ Creating VS Code settings..."
cat > .vscode/settings.json << 'EOL'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/venv": true,
        "**/.pytest_cache": true
    }
}
EOL

# Create launch.json for VS Code debugging
echo "🐞 Creating VS Code debug configuration..."
cat > .vscode/launch.json << 'EOL'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "RAG Demo",
            "type": "python", 
            "request": "launch",
            "program": "${workspaceFolder}/src/rag_braintrust_bot/rag_demo.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Ingest Documents",
            "type": "python",
            "request": "launch", 
            "program": "${workspaceFolder}/src/rag_braintrust_bot/ingestion/ingest.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
EOL

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local template..."
    cat > .env.local << 'EOL'
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
EOL
    echo "✅ Created .env.local template - please add your API keys!"
else
    echo "✅ .env.local already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. 📝 Edit .env.local and add your API keys:"
echo "   - VOYAGEAI_API_KEY"
echo "   - PINECONE_API_KEY" 
echo "   - BRAINTRUST_API_KEY"
echo ""
echo "2. 🗃️ Create Pinecone index named 'dev-rag-bot' with:"
echo "   - Dimension: 1024"
echo "   - Metric: cosine"
echo "   - Pod type: Serverless"
echo ""
echo "3. 📚 Ingest documents:"
echo "   source venv/bin/activate"
echo "   python src/rag_braintrust_bot/ingestion/ingest.py"
echo ""
echo "4. 🚀 Run RAG demo:"
echo "   python src/rag_braintrust_bot/rag_demo.py"
echo ""
echo "5. 🧪 Run evaluations:"
echo "   export \$(grep -v '^#' .env.local | xargs) && braintrust eval evals/eval_test.py"
echo ""
echo "Happy coding! 🚀" 