# Multimodal Document Chat System - Coding Test

## Project Overview

Build a system that allows users to upload PDF documents, extract text, images, and tables, and engage in multimodal chat based on the extracted content.

### Core Features
1. **Document Processing**: PDF parsing using Docling (extract text, images, tables)
2. **Vector Store**: Store extracted content in vector database
3. **Multimodal Chat**: Provide answers with related images/tables for text questions
4. **Multi-turn Conversation**: Maintain conversation context for continuous questioning

---

## Provided Components (Starting Point)

The following items are **already implemented and provided**:

### Infrastructure Setup
- Docker Compose configuration (PostgreSQL+pgvector, Redis, Backend, Frontend)
- Database schema and models (SQLAlchemy)
- API base structure (FastAPI)
- Frontend base structure (Next.js + TailwindCSS)

### Database Models
- `Document` - Uploaded document information
- `DocumentChunk` - Text chunks (with vector embeddings)
- `DocumentImage` - Extracted images
- `DocumentTable` - Extracted tables
- `Conversation` - Chat sessions
- `Message` - Chat messages

### API Endpoints (Skeleton provided)
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Document details
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/chat` - Send chat message
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation history

### Frontend Pages (Layout only)
- `/` - Home (document list)
- `/upload` - Document upload
- `/chat` - Chat interface
- `/documents/[id]` - Document details

### Development Tools
- FastAPI Swagger UI (`http://localhost:8000/docs`)
- Hot reload (Backend & Frontend)
- Environment configuration

---

## Core Features to Implement (Your Job)

You need to implement the following **3 core features**:

### 1. Document Processing Pipeline (Critical)

**Location**: `backend/app/services/document_processor.py`

**Requirements**:
```python
class DocumentProcessor:
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document to extract text, images, and tables.
        
        Implementation steps:
        1. Parse PDF using Docling
        2. Extract and chunk text (for vector storage)
        3. Extract and save images (filesystem + DB)
        4. Extract and save tables (structured data + image)
        5. Error handling and status updates
        
        Returns:
            {
                "status": "success",
                "text_chunks": 50,
                "images": 10,
                "tables": 5,
                "processing_time": 12.5
            }
        """
        pass
```

**Evaluation Criteria**:
- Docling integration and PDF parsing accuracy
- Image extraction and storage (filename, path, metadata)
- Table extraction (preserve structure, render as image)
- Text chunking strategy (chunk size, overlap)
- Error handling (invalid PDF, memory overflow, etc.)

---

### 2. Vector Store Integration (Critical)

**Location**: `backend/app/services/vector_store.py`

**Requirements**:
```python
class VectorStore:
    async def store_text_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: int
    ) -> int:
        """
        Store text chunks with vector embeddings.
        
        Implementation steps:
        1. Generate embeddings using OpenAI/HuggingFace
        2. Store in pgvector (vector + metadata)
        3. Include image/table references in metadata
        
        Returns:
            Number of stored chunks
        """
        pass
    
    async def search_similar(
        self, 
        query: str, 
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
        pass
```

**Evaluation Criteria**:
- Embedding model selection and integration
- pgvector utilization (cosine similarity, indexing)
- Metadata management (image/table references)
- Search accuracy and performance

---

### 3. Multimodal Chat Engine (Critical)

**Location**: `backend/app/services/chat_engine.py`

**Requirements**:
```python
class ChatEngine:
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process user message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (multi-turn support)
        2. Find relevant context using vector search
        3. Find related images/tables
        4. Generate answer using LLM
        5. Include image/table URLs in response
        
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/abc123.png",
                        "caption": "Figure 1: ..."
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/xyz789.png",
                        "caption": "Table 1: ..."
                    }
                ],
                "processing_time": 2.5
            }
        """
        pass
```

**Evaluation Criteria**:
- RAG implementation quality (relevance, accuracy)
- Multi-turn conversation support (context maintenance)
- Include images/tables in responses
- LLM prompt engineering
- Response speed and user experience

---

## System Architecture

```
┌─────────────┐
│   Frontend  │ (Next.js)
│  Chat UI    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Backend   │ (FastAPI)
│  API Server │
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Document   │   │    Chat     │
│  Processor  │   │   Engine    │
│  (Docling)  │   │   (RAG)     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────┐
│      Vector Store           │
│    (PostgreSQL+pgvector)    │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│    File Storage             │
│  (Images, Tables, PDFs)     │
└─────────────────────────────┘
```

---

## Data Models

### Document
```python
class Document:
    id: int
    filename: str
    file_path: str
    upload_date: datetime
    processing_status: str  # 'pending', 'processing', 'completed', 'error'
    total_pages: int
    text_chunks_count: int
    images_count: int
    tables_count: int
```

### DocumentChunk
```python
class DocumentChunk:
    id: int
    document_id: int
    content: str
    embedding: Vector(1536)  # pgvector
    page_number: int
    chunk_index: int
    metadata: JSON  # {related_images: [...], related_tables: [...]}
```

### DocumentImage
```python
class DocumentImage:
    id: int
    document_id: int
    file_path: str
    page_number: int
    caption: str
    width: int
    height: int
```

### DocumentTable
```python
class DocumentTable:
    id: int
    document_id: int
    image_path: str  # Rendered table as image
    data: JSON  # Structured table data
    page_number: int
    caption: str
```

### Conversation & Message
```python
class Conversation:
    id: int
    title: str
    created_at: datetime
    document_id: Optional[int]  # Conversation about specific document

class Message:
    id: int
    conversation_id: int
    role: str  # 'user', 'assistant'
    content: str
    sources: JSON  # Sources used in answer (text, images, tables)
    created_at: datetime
```

---

## Tech Stack

### Backend
- **Framework**: FastAPI
- **PDF Processing**: Docling
- **Vector DB**: PostgreSQL + pgvector
- **Embeddings**: HuggingFace Sentence Transformers (Free)
- **LLM**: Ollama (Free, Local)
- **Task Queue**: Celery + Redis (optional)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: TailwindCSS
- **UI Components**: shadcn/ui
- **State Management**: React Hooks
- **API Client**: fetch/axios

### Infrastructure
- **Database**: PostgreSQL 15 + pgvector
- **Cache**: Redis
- **Container**: Docker + Docker Compose

---

## LLM Setup (Ollama - Recommended)

**This project uses Ollama as the default LLM provider - completely free and runs locally!**

### Step-by-Step Ollama Setup

#### 1. Install Ollama

**For macOS:**
```bash
brew install ollama
```

**For Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**For Windows:**
- Download installer from [ollama.com/download](https://ollama.com/download)
- Run the installer

#### 2. Start Ollama Service

```bash
# Start Ollama (runs in background)
ollama serve
```

#### 3. Download Recommended Model

```bash
# Download Llama 3.2 (3B - fast and efficient)
ollama pull llama3.2

# Verify installation
ollama list
```

#### 4. Test Your Setup

```bash
# Quick test
ollama run llama3.2 "Hello, tell me about AI"
```

#### 5. Environment Configuration

Your `.env` file should already be configured for Ollama:

```bash
# LLM Configuration (Default: Ollama)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional: OpenAI (if you prefer to use paid API)
# OPENAI_API_KEY=your-openai-api-key
```

### Model Options

| Model | Size | RAM Required | Speed | Quality | Best For |
|-------|------|--------------|--------|---------|----------|
| `llama3.2` | 3B | 4GB+ | Fast | Good | **Development (Recommended)** |
| `llama3.1` | 8B | 8GB+ | Medium | Very Good | Production |
| `mistral` | 7B | 8GB+ | Medium | Good | Balanced |
| `phi3` | 3.8B | 4GB+ | Fast | Good | Code tasks |

### Troubleshooting Ollama

**Problem**: `ollama: command not found`
```bash
# Check if Ollama is in PATH
which ollama

# If not installed, reinstall:
curl -fsSL https://ollama.com/install.sh | sh
```

**Problem**: Model download fails
```bash
# Check internet connection and retry
ollama pull llama3.2 --verbose
```

**Problem**: Out of memory
```bash
# Use smaller model
ollama pull phi3

# Update .env
OLLAMA_MODEL=phi3
```

**Problem**: Can't connect to Ollama
```bash
# Make sure service is running
ollama serve

# Check if port 11434 is available
curl http://localhost:11434/api/tags
```

---

## Alternative LLM Options (Optional)

If you prefer cloud APIs or have issues with Ollama:

### Option A: OpenAI (Paid)
```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
```

### Option B: Google Gemini (Free Tier)
```bash
# .env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-api-key
```

### Option C: Groq (Free Tier)
```bash
# .env
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
```

**Default Recommendation**: Stick with **Ollama** - it's free, private, and has no rate limits!

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- Ollama (for LLM - see setup above)

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd coding-test-4th

# 2. Install and setup Ollama (see LLM Setup section above)
ollama pull llama3.2

# 3. Set up environment
cp .env.example .env
# .env is already configured for Ollama - no changes needed!

# 4. Start all services
docker-compose up -d

# 5. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## Evaluation Criteria (100 points)

### 1. Code Quality (25 points)
- **Structure** (10 points): Module separation, responsibility separation, reusability
- **Readability** (8 points): Naming, comments, code style
- **Error Handling** (7 points): Exception handling, error messages, recovery strategy

### 2. Feature Implementation (40 points)
- **Document Processing** (15 points):
  - Docling integration and PDF parsing (5 points)
  - Image extraction and storage (5 points)
  - Table extraction and storage (5 points)

- **Vector Store** (10 points):
  - Embedding generation and storage (5 points)
  - Similarity search accuracy (5 points)

- **Chat Engine** (15 points):
  - RAG implementation quality (5 points)
  - Multimodal responses (images/tables included) (5 points)
  - Multi-turn conversation support (5 points)

### 3. UX/UI (15 points)
- **Chat Interface** (8 points): Intuitiveness, responsiveness, image/table display
- **Document Upload/Management** (4 points): Progress indication, error display
- **Design** (3 points): Consistency, aesthetics

### 4. Documentation (10 points)
- **README** (4 points): Installation, execution, feature explanation
- **Code Comments** (3 points): Complex logic explanation
- **API Documentation** (3 points): Swagger or separate documentation

### 5. Testing (10 points)
- **Unit Tests** (5 points): Core logic testing
- **Integration Tests** (3 points): API endpoint testing
- **Test Coverage** (2 points): 60% or higher

---

## Bonus Points (+20 points)

- **Advanced PDF Processing** (+5 points): OCR, complex layout handling
- **Multi-document Search** (+5 points): Search across multiple documents
- **Real-time Chat** (+5 points): WebSocket-based
- **Deployment** (+5 points): Production deployment setup (Railway, Vercel, etc.)

---

## Submission Requirements

### What to Submit
1. **GitHub Repository** (public or private with access)
2. **Complete source code** (backend + frontend)
3. **Docker configuration** (docker-compose.yml)
4. **Documentation** (README, API docs, architecture)
5. **Sample data** (at least one test PDF)

### README Must Include
- Project overview
- Tech stack
- Setup instructions (Docker + Ollama)
- Environment variables (.env.example)
- API testing examples
- Features implemented
- Known limitations
- Future improvements
- Screenshots (minimum 5):
  - Document upload screen
  - Document processing completion screen
  - Chat interface
  - Answer example with images
  - Answer example with tables

### How to Submit
1. Push code to GitHub
2. Test that `docker-compose up` works with Ollama
3. Send repository URL via email
4. Include any special instructions

---

## Test Scenarios

### Scenario 1: Basic Document Processing
1. Upload a technical paper PDF
2. Verify text, images, and tables extraction
3. Check extracted content on document detail page

### Scenario 2: Text-based Question
1. Ask "What is the main conclusion of this paper?"
2. Verify answer is generated with relevant text context

### Scenario 3: Image-related Question
1. Ask "Show me the architecture diagram"
2. Verify related images are displayed in chat

### Scenario 4: Table-related Question
1. Ask "What are the experimental results?"
2. Verify related tables are displayed in chat

### Scenario 5: Multi-turn Conversation
1. First question: "What is the dataset used?"
2. Follow-up: "How many samples does it contain?"
3. Verify previous conversation context is maintained

---

## Sample PDF

A sample PDF file is provided: `1706.03762v7.pdf`

This is a technical paper ("Attention Is All You Need") with:
- Multiple pages with text content
- Diagrams and architecture figures
- Tables with experimental results
- Complex layouts for testing

You should use this PDF to test your implementation.

---

## Implementation Guidelines

Refer to the service skeleton files for detailed implementation guidance:
- `backend/app/services/document_processor.py` - Document processing guidelines
- `backend/app/services/vector_store.py` - Vector store implementation tips
- `backend/app/services/chat_engine.py` - Chat engine implementation tips

Each file contains detailed TODO comments with implementation hints and examples.

---

## Troubleshooting

### Ollama Issues
**Problem**: Ollama model not responding
**Solution**: 
- Check if Ollama service is running: `ollama serve`
- Verify model is downloaded: `ollama list`
- Test with simple query: `ollama run llama3.2 "hello"`

### Document Processing Issues
**Problem**: Docling can't extract tables
**Solution**: 
- Check PDF format (ensure it's not scanned image)
- Add fallback parsing logic
- Manually define table structure patterns

### Vector Search Issues
**Problem**: Search results are not relevant
**Solution**:
- Verify embedding model is working
- Check chunk size and overlap settings
- Ensure pgvector extension is installed
- Test with simple queries first

### CORS Issues
**Problem**: Frontend can't call backend API
**Solution**:
- Add CORS middleware in FastAPI
- Allow origin: http://localhost:3000
- Check network configuration in Docker

---

## FAQ

**Q: Ollama won't start.**
A: Try `ollama serve` in a separate terminal and check if port 11434 is available.

**Q: Docling won't install.**
A: Try `pip install docling` or use the Docker image.

**Q: Where should I save images?**
A: Save to `backend/uploads/images/` directory and store only the path in DB.

**Q: How should I display tables?**
A: Render tables as images or display JSON data as HTML tables in frontend.

**Q: How do I test the system locally?**
A: Follow the Getting Started section and use the provided sample PDF (1706.03762v7.pdf).

**Q: Can I use a different LLM?**
A: Yes! The system supports OpenAI, Gemini, and Groq. See Alternative LLM Options section.

---

## Questions?

If you have any questions, please create an issue or contact us via email.

Good luck!

---

## Tips for Success

1. **Start Simple**: Get core features working before adding advanced features
2. **Test Early**: Test document processing with sample PDF immediately
3. **Use Ollama**: Free, no API keys needed, perfect for development
4. **Focus on Core**: Perfect the RAG pipeline first
5. **Document Well**: Clear README helps evaluators understand your work
6. **Handle Errors**: Graceful error handling shows maturity
7. **Ask Questions**: If requirements are unclear, document your assumptions

---

## Support

For questions about this coding challenge:
- Open an issue in this repository
- Email: recruitment@interopera.co

---

**Version**: 1.1 (Updated for Ollama)  
**Last Updated**: 2025-12-20  
**Author**: InterOpera-Apps Hiring Team
