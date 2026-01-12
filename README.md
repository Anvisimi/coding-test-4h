# Implementation Documentation

## Project Overview

This implementation delivers a **Multimodal Document Chat System** that processes PDF documents and enables intelligent Q&A conversations. The system extracts text, images, and tables from PDFs, stores them with vector embeddings for semantic search, and provides contextually relevant answers using RAG (Retrieval-Augmented Generation) with multimodal support.

**Core Value Proposition:**
- Upload PDF documents and automatically extract structured content
- Ask questions in natural language and receive answers with supporting visual evidence
- Maintain conversation context across multiple turns
- Support for multiple LLM providers (OpenAI, Ollama, Gemini, Groq)

---

## Tech Stack

**Backend:**
- FastAPI (Python 3.11+) - REST API framework
- Docling - PDF parsing and content extraction
- PostgreSQL 15 + pgvector - Vector database for semantic search
- SQLAlchemy - ORM for database operations
- OpenAI API / HuggingFace - Embedding generation
- Multiple LLM providers (OpenAI, Ollama, Gemini, Groq)

**Frontend:**
- Next.js 14 (App Router) - React framework
- TailwindCSS - Styling
- shadcn/ui - UI component library

**Infrastructure:**
- Docker & Docker Compose - Containerization
- pytest - Testing framework (61% code coverage)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                       │
│              (Next.js 14 + TailwindCSS)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Upload   │  │ Documents│  │   Chat   │               │
│  │   Page   │  │   List   │  │ Interface│               │
│  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP/REST API
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Backend API Layer                     │
│                    (FastAPI)                            │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ Documents API    │  │   Chat API       │             │
│  │ - Upload         │  │   - Send Message │             │
│  │ - List/Get       │  │   - Conversations│             │
│  │ - Delete         │  │   - History      │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
└───────────┼─────────────────────┼───────────────────────┘
            │                     │
            ▼                     ▼
┌──────────────────┐    ┌──────────────────────────────┐
│ Document         │    │      Chat Engine             │
│ Processor        │    │   (RAG: Retrieval-Augmented  │
│                  │    │        Generation)           │
│ Orchestrates:    │    │  ┌──────────────────────┐    │
│ - Docling Parser │    │  │ Load Conversation    │    │
│ - Extractors     │    │  │ History              │    │
│ - Validators     │    │  └──────────┬───────────┘    │
└──────┬───────────┘    │             │                │
       │                │             ▼                │
       │                │  ┌──────────────────────┐    │
       ├────────────────┼─▶│ Vector Search        │    │
       │                │  │ (Semantic Similarity)│    │
       │                │  │ (RAG Retrieval)      │    │
       │                │  └──────────┬───────────┘    │
       │                │             │                │
       │                │             ▼                │
       │                │  ┌──────────────────────┐    │
       │                │  │ Find Related Media   │    │
       │                │  │ (Images/Tables)      │    │
       │                │  └──────────┬───────────┘    │
       │                │             │                │
       │                │             ▼                │
       │                │  ┌──────────────────────┐    │
       │                │  │ RAG Generation       │    │
       │                │  │ (Context + History)  │    │
       │                │  │ → LLM: OpenAI        │    │
       │                │  │   GPT-4o-mini        │    │
       │                │  └──────────────────────┘    │
       │                └──────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│              Extraction & Processing Layer               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Text         │  │ Image        │  │ Table        │    │
│  │ Extractor    │  │ Extractor    │  │ Extractor    │    │
│  │              │  │              │  │              │    │
│  │ - Chunking   │  │ - Docling    │  │ - Structure  │    │
│  │ - Overlap    │  │ - PyMuPDF    │  │ - Rendering  │    │
│  │ - Metadata   │  │ - Metadata   │  │ - Metadata   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │            │
│         └─────────┬───────┴─────────┬───────┘            │
│                   │                 │                    │
│                   ▼                 ▼                    │
│         ┌──────────────────┐  ┌──────────────────┐       │
│         │ Binary Validator │  │ Vector Store     │       │
│         │                  │  │                  │       │
│         │ - Validate text  │  │ - Embeddings:    │       │
│         │ - Filter binary  │  │   OpenAI         │       │
│         │ - Clean data     │  │   text-embedding-│       │
│         └──────────────────┘  │   3-small (1536) │       │
│                               │ - Storage        │       │
│                               │ - Search         │       │
│                               └─────────┬────────┘       │
└─────────────────────────────────────────┼────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
    ┌────────────────────── ┐  ┌──────────────────────┐  ┌──────────────────────┐
    │   PostgreSQL +        │  │    File Storage      │  │   LLM Provider       │
    │   pgvector            │  │                      │  │                      │
    │                       │  │  - PDFs              │  │  - OpenAI            │
    │  - Document Metadata  │  │  - Images            │  │    (GPT-4o-mini)     │
    │  - Text Chunks        │  │  - Tables            │  │                      │
    │  - Vector Embeddings  │  │                      │  │                      │
    │  - Images Metadata    │  │                      │  │                      │
    │  - Tables Metadata    │  │                      │  │                      │
    │  - Conversations      │  │                      │  │                      │
    │  - Messages           │  │                      │  │                      │
    └───────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

**Key Components:**

1. **Frontend Layer**: Next.js application with upload, document list, and chat interfaces
2. **API Layer**: FastAPI REST endpoints for document management and chat
3. **Document Processor**: Orchestrates PDF parsing using Docling and coordinates extraction
4. **Extractors**: Specialized modules for text, image, and table extraction
   - **TextExtractor**: Handles text chunking with overlap and metadata
   - **ImageExtractor**: Extracts images from Docling and PyMuPDF with fallback
   - **TableExtractor**: Extracts structured tables and renders them as images
5. **Binary Validator**: Validates and filters binary data from text extraction
6. **Chat Engine**: Implements RAG (Retrieval-Augmented Generation) pipeline with conversation history, semantic search, and LLM integration
7. **Vector Store**: Manages embeddings using OpenAI text-embedding-3-small (1536 dimensions) and similarity search using pgvector
8. **Database**: PostgreSQL with pgvector extension for vector operations and metadata storage
9. **File Storage**: Filesystem storage for PDFs, extracted images, and table renders
10. **LLM Provider**: OpenAI GPT-4o-mini for response generation

---

## Setup Instructions (Docker)

### Prerequisites
- Docker & Docker Compose installed
- API key for chosen LLM provider (see Environment Variables)

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd coding-test-4h

# 2. Configure environment
cd backend
cp .env.example .env
# Edit .env with your API keys (see Environment Variables section)

# 3. Start services
docker-compose up -d

# 4. Verify services
docker-compose ps
```

**Access Points:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**Stop Services:**
```bash
docker-compose down
```

---

## Environment Variables

Create `.env` file in `backend/` directory:

```bash
# Database
DATABASE_URL=postgresql://docuser:docpass@localhost:5432/docdb

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# LLM Provider (choose one: openai, ollama, gemini, groq)
LLM_PROVIDER=openai

# OpenAI (if LLM_PROVIDER=openai)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama (if LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Gemini (if LLM_PROVIDER=gemini)
GOOGLE_API_KEY=your-google-api-key

# Groq (if LLM_PROVIDER=groq)
GROQ_API_KEY=your-groq-api-key

# Upload Configuration
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800  # 50 MB

# Vector Store Configuration
EMBEDDING_DIMENSION=1536
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

**Note:** Only configure the LLM provider you intend to use. See README.md "Free LLM Options" section for free alternatives.

---

## API Testing Examples

All examples use the test PDF: `1706.03762v7.pdf` ("Attention Is All You Need" paper)

**Download test PDF:**
```bash
curl -o 1706.03762v7.pdf https://arxiv.org/pdf/1706.03762.pdf
```

### Example 1: Upload Document

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@1706.03762v7.pdf"
```

**Response:**
```json
{
  "id": 1,
  "filename": "1706.03762v7.pdf",
  "status": "pending",
  "message": "Document uploaded successfully. Processing will begin shortly."
}
```

### Example 2: Check Processing Status

```bash
curl "http://localhost:8000/api/documents/1"
```

**Response:**
```json
{
  "id": 1,
  "filename": "1706.03762v7.pdf",
  "status": "completed",
  "total_pages": 15,
  "text_chunks_count": 87,
  "images_count": 6,
  "tables_count": 4,
  "upload_date": "2025-01-15T10:30:00"
}
```

### Example 3: Text-based Question

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main contribution of this paper?",
    "document_id": 1
  }'
```

**Response:**
```json
{
  "conversation_id": 1,
  "message_id": 1,
  "answer": "The main contribution of this paper is the introduction of the Transformer architecture, which relies entirely on attention mechanisms, eliminating the need for recurrence and convolutions...",
  "sources": [
    {
      "type": "text",
      "content": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...",
      "page": 1,
      "score": 0.95
    }
  ],
  "processing_time": 2.3
}
```

### Example 4: Image-related Question

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me the Transformer architecture diagram",
    "conversation_id": 1
  }'
```

**Response:**
```json
{
  "conversation_id": 1,
  "message_id": 2,
  "answer": "Here is the Transformer architecture diagram from the paper. The model consists of an encoder and decoder stack...",
  "sources": [
    {
      "type": "image",
      "url": "/uploads/images/figure1.png",
      "caption": "Figure 1: The Transformer - model architecture",
      "page": 2
    },
    {
      "type": "text",
      "content": "The Transformer follows this architecture with stacked self-attention and point-wise, fully connected layers...",
      "page": 2,
      "score": 0.92
    }
  ],
  "processing_time": 1.8
}
```

### Example 5: Table-related Question

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the BLEU scores for different models?",
    "conversation_id": 1
  }'
```

**Response:**
```json
{
  "conversation_id": 1,
  "message_id": 3,
  "answer": "According to the paper, the BLEU scores for different models on WMT 2014 English-to-German translation are shown in the following table...",
  "sources": [
    {
      "type": "table",
      "url": "/uploads/tables/table1.png",
      "caption": "Table 1: BLEU scores on WMT 2014 English-to-German translation",
      "page": 3,
      "data": {
        "rows": [
          ["Model", "BLEU"],
          ["Transformer (base)", "28.4"],
          ["Transformer (big)", "29.3"]
        ]
      }
    }
  ],
  "processing_time": 2.1
}
```

### Example 6: Multi-turn Conversation

```bash
# First question
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What attention mechanism does this paper propose?",
    "conversation_id": 1
  }'

# Follow-up (uses conversation history)
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does it compare to RNN and CNN?",
    "conversation_id": 1
  }'
```

---

## Features Implemented

### Document Processing Pipeline

**PDF Parsing & Extraction:**
- Docling integration for robust PDF parsing
- Text extraction with intelligent chunking (1000 chars, 200 overlap)
- Image extraction with metadata (caption, dimensions, page number)
- Table extraction as structured JSON + rendered images
- Background processing with status tracking (pending → processing → completed/error)

**Error Handling:**
- Invalid file type validation
- File size limit enforcement (50 MB)
- Graceful error recovery with status updates
- Retry mechanisms for transient failures

### Vector Store Integration

**Embedding Generation:**
- OpenAI API (text-embedding-3-small, 1536 dimensions) - primary
- HuggingFace Sentence Transformers - fallback option

**Search & Retrieval:**
- PostgreSQL + pgvector for efficient similarity search
- Cosine similarity with configurable top-k results
- Metadata management (image/table references in chunk metadata)
- Related content retrieval by page number or metadata

### Multimodal Chat Engine

**RAG Implementation:**
- Retrieval-Augmented Generation with context-aware responses
- Multi-turn conversation support (maintains history)
- Vector-based semantic search for relevant chunks
- Multimodal responses (includes related images and tables)

**LLM Provider Support:**
- OpenAI GPT-4o-mini (default, best quality)
- Ollama (local, free, no rate limits)
- Google Gemini (free tier, 60 req/min)
- Groq (free tier, very fast)

**Response Format:**
- Structured answers with source attribution
- Source types: text, image, table
- Page numbers and relevance scores included

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/documents/upload` | POST | Upload PDF document |
| `/api/documents` | GET | List documents (paginated) |
| `/api/documents/{id}` | GET | Get document details |
| `/api/documents/{id}` | DELETE | Delete document |
| `/api/chat` | POST | Send chat message |
| `/api/conversations` | GET | List conversations |
| `/api/conversations/{id}` | GET | Get conversation history |

### Testing

- **Overall Coverage:** 77%
- **Unit Test Coverage:** 74% (115 tests)
- **Integration Test Coverage:** 29% (10 tests)
- **Coverage Areas:**
  - Document processing logic
  - Vector store operations
  - Chat engine functionality
  - API error handling
  - LLM provider integration

**Running Tests:**
```bash
# Run all tests (unit + integration)
cd backend
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage report
pytest --cov=app --cov-config=.coveragerc --cov-report=term
```

---

## Known Limitations

1. **PDF Processing**
   - Optimized for text-based PDFs; scanned PDFs have limited accuracy
   - Complex multi-column layouts may require chunking parameter tuning
   - Large PDFs (>100 pages) have longer processing times

2. **Search Capabilities**
   - Single document search only (multi-document search not implemented)
   - Search accuracy depends on embedding model quality
   - Chunk boundaries may split related content

3. **Performance**
   - Synchronous document processing (no background queue)
   - No caching for repeated queries
   - Vector search performance degrades with very large collections

4. **LLM Integration**
   - Response quality varies by provider
   - Free providers have rate limits
   - Long conversations may exceed context windows

5. **Frontend**
   - Basic UI implementation

---

## Future Improvements

1. **Enhanced Processing**
   - OCR support for scanned PDFs
   - Better complex layout handling
   - Support for DOCX, HTML formats

2. **Advanced Search**
   - Multi-document search
   - Hybrid search (keyword + semantic)
   - Advanced query understanding

3. **Performance**
   - Query caching
   - Background job queue (Celery)
   - Database optimization

4. **User Experience**
   - WebSocket-based real-time updates
   - Better error messages
   - Document preview
   - Conversation export

5. **LLM Enhancements**
   - Streaming responses
   - Custom prompt templates
   - Multi-modal LLM support

6. **Production Readiness**
   - User authentication
   - Rate limiting
   - Document access control
   - Horizontal scaling

---

## Screenshots

All screenshots demonstrate the system using `1706.03762v7.pdf` ("Attention Is All You Need" paper).

### 1. Document Upload Screen
![Document Upload](screenshots/01-upload-screen.png)
![Document Upload](screenshots/02-upload-screen.png)
*Uploading the test PDF file through the web interface*

### 2. Document Processing Completion
![Processing Complete](screenshots/03-processing-complete.png)
*Processing results showing extraction statistics:
- Text chunks: 83
- Images: 6 figures
- Tables: 4 tables
- Status: Completed*

### 3. Chat Interface
![Chat Interface](screenshots/04-chat-interface.png)
*Chat interface with sample questions about the Transformer paper*

### 4. Answer with Images (Transformer Architecture Diagram)
![Answer with Images](screenshots/05-answer-with-images.png)
*Chat response showing Figure 1 (Transformer architecture diagram) along with text explanation*

### 5. Answer with Tables (BLEU Score Comparisons)
![Answer with Tables](screenshots/06-answer-with-tables.png)
![Answer with Tables](screenshots/07-answer-with-tables.png)
*Chat response displaying performance comparison tables with BLEU scores*

### 6. Multi-turn Conversation
![Multi-turn Conversation](screenshots/09-multiturn-question.png)
![Multi-turn Conversation](screenshots/08-multiturn-question.png)
*Chat response displaying performance comparison tables with BLEU scores*

### 7. Specific Technical Question
![Specific Technical Question](screenshots/10-tech-question.png)
*Chat response displaying performance comparison tables with BLEU scores*

---
