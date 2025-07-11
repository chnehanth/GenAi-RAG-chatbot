# GenAi-RAG-chatbot
A Streamlit-based chatbot that allows you to interact with the contents of a PDF using Retrieval Augmented Generation (RAG) techniques and vector search.

## Features
- Upload and chat with your own PDF documents
- Uses embeddings and vector database for context-aware answers
- Simple, interactive web UI with Streamlit

## How it works
1. The PDF is loaded and split into chunks.
2. Each chunk is embedded using a language model.
3. Chunks are stored in a vector database (ChromaDB).
4. When you ask a question, the most relevant chunks are retrieved and used to generate an answer.


## Project Structure
- `app.py` - Main Streamlit app
- `utils/` - Helper modules for loading, embedding, retrieval, and QA
- `requirements.txt` - Python dependencies
- `data/` - Place your PDF files here
- `chroma_db/` - Vector database files (auto-generated)
