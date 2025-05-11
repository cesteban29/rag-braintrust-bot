#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up development environment..."

# Use Python 3.11 specifically
PYTHON_CMD="/opt/homebrew/opt/python@3.11/bin/python3.11"
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python 3.11 not found. Please install it with: brew install python@3.11"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create .vscode directory if it doesn't exist
if [ ! -d ".vscode" ]; then
    echo "📁 Creating .vscode directory..."
    mkdir .vscode
fi

# Create settings.json for VS Code
echo "⚙️ Creating VS Code settings..."
cat > .vscode/settings.json << EOL
{
    "python.defaultInterpreterPath": "\${workspaceFolder}/venv/bin/python",
    "python.analysis.extraPaths": ["\${workspaceFolder}"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
EOL

# Create launch.json for VS Code debugging
echo "🐞 Creating VS Code debug configuration..."
cat > .vscode/launch.json << EOL
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "\${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
EOL

echo "✅ Setup complete! Don't forget to:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Create a .env.local file with your API keys"
echo "3. Run the ingest script: python ingest.py" 