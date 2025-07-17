import braintrust
import json

# Create a new project in Braintrust using the environment variable BRAINTRUST_PROJECT_NAME
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

project_name = os.getenv('BRAINTRUST_PROJECT_NAME', 'rag-braintrust-bot')
project = braintrust.projects.create(name=project_name)

# Define a multi-turn conversation prompt with tool calls
multi_turn_conversation = project.prompts.create(
    name="MultiTurnConversation",
    slug="multi-turn-conversation",
    description="Simulate a multi-turn conversation with document retrieval about Braintrust",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant with access to a document retrieval tool. You can use this tool to find relevant information about Braintrust from the knowledge base."},
        {"role": "user", "content": "What can you tell me about Braintrust?"},
        {"role": "assistant", "content": "Let me search the knowledge base for information about Braintrust."},
        {"role": "assistant", "content": "I'll use the document retrieval tool to find relevant information.", "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "bt-get-documents-rag",
                    "arguments": "{\"query\": \"What is Braintrust and how does it work?\"}"
                }
            }
        ]},
        {"role": "tool", "content": "Found relevant documents about Braintrust.", "tool_call_id": "call_123", "documents": [
            {
                "title": "Introduction to Braintrust",
                "content": "Braintrust is a platform for building and managing AI applications, including tools for prompt engineering, document retrieval, and more.",
                "id": "doc1",
                "score": 0.92,
                "tags": ["braintrust", "introduction"]
            }
        ]},
        {"role": "assistant", "content": "Based on the retrieved information, Braintrust is a platform for building and managing AI applications, including tools for prompt engineering, document retrieval, and more. Would you like to know more about how it's implemented?"},
        {"role": "user", "content": "Yes, how is it implemented?"},
        {"role": "assistant", "content": "I'll search for implementation details."},
        {"role": "assistant", "content": "I'll use the document retrieval tool to find implementation details.", "tool_calls": [
            {
                "id": "call_124",
                "type": "function",
                "function": {
                    "name": "bt-get-documents-rag",
                    "arguments": {"query": "How is Braintrust implemented? Implementation details and architecture"}
                }
            }
        ]},
        {"role": "tool", "content": "Found implementation details.", "tool_call_id": "call_124", "documents": [
            {
                "title": "Braintrust Implementation",
                "content": "Braintrust implementation typically involves a vector database for document storage, an embedding model for semantic search, and a language model for generating responses based on retrieved context.",
                "id": "doc2",
                "score": 0.89,
                "tags": ["implementation", "architecture"]
            }
        ]},
        {"role": "assistant", "content": "According to the documentation, Braintrust implementation typically involves three main components: a vector database for document storage, an embedding model for semantic search, and a language model for generating responses based on retrieved context. Would you like me to explain any of these components in more detail?"}
    ]
)

# Log the creation of the prompt
print(f"Multi-turn conversation prompt created with ID: {multi_turn_conversation.id}")



