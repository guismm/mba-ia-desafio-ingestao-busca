# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system for PDF document ingestion and question-answering using LangChain with Google Gemini models and PGVector for vector storage.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL with pgVector
docker compose up -d

# Ingest a PDF document into vector store
python src/ingest.py

# Start interactive chat (queries the ingested documents)
python src/chat.py

# Enable debug mode to see the full prompt with context
DEBUG_PROMPT=1 python src/chat.py
```

## Architecture

The system follows a standard RAG pipeline:

1. **Ingestion** (`src/ingest.py`): Loads PDF, splits into chunks (1000 chars, 150 overlap) with `RecursiveCharacterTextSplitter`, injects table headers for context, stores embeddings in PGVector
2. **Search** (`src/search.py`): Builds a LangChain RAG chain that retrieves top-10 relevant chunks and generates answers via LLM
3. **Chat** (`src/chat.py`): Interactive REPL that uses the search chain

## Configuration

Copy `.env.example` to `.env` and configure:
- `GOOGLE_API_KEY`: Google API key for Gemini access
- `GOOGLE_EMBEDDING_MODEL`: Embedding model (default: `models/embedding-001`)
- `GOOGLE_LLM_MODEL`: Chat model (default: `gemini-2.5-flash-lite`)
- `PGVECTOR_URL`: PostgreSQL connection string
- `PGVECTOR_COLLECTION`: Collection name in PGVector
- `PDF_PATH`: Path to PDF for ingestion

## Dependencies

- Requires Google API key with Gemini access
- Requires PostgreSQL with pgvector extension (provided via docker-compose)
