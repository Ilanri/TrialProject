# Streamlit RAG Chatbot

This project is a minimal Streamlit chatbot app using Retrieval-Augmented Generation (RAG) with Groq as the LLM and FAISS as the vector store for document retrieval.

## Features

- Chatbot interface powered by Streamlit
- RAG architecture: retrieves relevant context from your documents
- Uses Groq for LLM responses
- FAISS for fast vector search

## Setup

Use source .venv/bin/activate

1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Configuration

- Set your Groq API key in the environment variable `GROQ_API_KEY`.
- Place your knowledge base files in the `data/` directory.

 If you update intro.txt, simply delete persona_prompt.txt to force a new persona to be generated.
