# Multimodal Document Chat System - Coding Test

## Project Overview

Build a system that allows users to upload PDF documents, extract text, images, and tables, and engage in multimodal chat based on the extracted content.

### Reference Materials

This coding test is inspired by modern AI document processing capabilities. For background understanding of the underlying AI principles, you may reference these official technical documents:

- **Anthropic Claude 3 Model Card**: https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf
- **Claude Opus 4 & Sonnet 4 System Card**: https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf

These documents provide insights into:
- Multimodal AI capabilities (text + vision)
- Document understanding and processing approaches
- AI safety considerations in production systems
- Constitutional AI principles for responsible AI development

**Note**: The task below is an independent implementation challenge and does not require reproducing any content from these reference materials.

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
- **Embeddings**: OpenAI API or HuggingFace
- **LLM**: OpenAI GPT-4o-mini or Ollama
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

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+
- OpenAI API Key (or Ollama for local LLM)

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd coding-test-4th

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start all services
docker-compose up -d

# 4. Access the application
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
5. **Test Results**: Screenshots and examples using the provided `1706.03762v7.pdf` file

### README Must Include
- Project overview
- Tech stack
- Setup instructions (Docker)
- Environment variables (.env.example)
- API testing examples **using the provided PDF**
- Features implemented
- Known limitations
- Future improvements
- **Screenshots (minimum 5) using `1706.03762v7.pdf`**:
  - Document upload screen with the test PDF
  - Document processing completion screen showing extraction results
  - Chat interface with sample questions
  - Answer example with images (Transformer architecture diagram)
  - Answer example with tables (BLEU score comparisons)

### Demo Requirements
Your submission should include working examples with the "Attention Is All You Need" paper:

1. **Document Processing Demo**: Show successful extraction of text, images, and tables
2. **Chat Examples**: Include at least 3 working Q&A examples using the test PDF
3. **Multimodal Responses**: Demonstrate answers that include both text and visual elements
4. **Error Handling**: Show how your system handles edge cases

**Note**: All evaluators will test your system using the same PDF to ensure fair and consistent assessment.

### How to Submit
1. Push code to GitHub
2. Test that `docker-compose up` works
3. Send repository URL via email
4. Include any special instructions

---

## Test Scenarios

Use the provided "Attention Is All You Need" paper (`1706.03762v7.pdf`) for these test scenarios:

### Scenario 1: Basic Document Processing
1. Download and upload `1706.03762v7.pdf`
2. Verify text, images, and tables extraction
3. Check extracted content on document detail page
4. **Expected Results**:
   - Text chunks: ~80-100 chunks
   - Images: ~6 figures (architecture diagrams)
   - Tables: ~4-6 tables (performance results)

### Scenario 2: Text-based Question
1. Ask: **"What is the main contribution of this paper?"**
2. **Expected Answer**: Should mention Transformer architecture and attention mechanisms
3. Verify answer includes relevant text context from abstract/conclusion

### Scenario 3: Image-related Question
1. Ask: **"Show me the Transformer architecture diagram"**
2. **Expected Result**: Should display Figure 1 (Transformer model architecture)
3. Verify the architecture diagram image is included in chat response

### Scenario 4: Table-related Question
1. Ask: **"What are the BLEU scores for different models?"**
2. **Expected Result**: Should display performance comparison tables
3. Verify tables with BLEU scores are shown in chat

### Scenario 5: Multi-turn Conversation
1. First question: **"What attention mechanism does this paper propose?"**
2. Follow-up: **"How does it compare to RNN and CNN?"**
3. **Expected Behavior**: Second answer should reference the first question's context
4. Verify conversation history is maintained

### Scenario 6: Specific Technical Question
1. Ask: **"What is the computational complexity of self-attention?"**
2. **Expected Answer**: Should mention O(n²·d) complexity and reference Section 3.2.1
3. Verify answer includes mathematical details from the paper

### Additional Test Questions to Try

**Architecture Questions**:
- "Explain the encoder-decoder structure"
- "What are the different types of attention used?"

**Performance Questions**:
- "How does the model perform on WMT translation tasks?"
- "What are the training details and hyperparameters?"

**Technical Questions**:
- "What is positional encoding and why is it needed?"
- "How many attention heads are used in the base model?"

---

## Sample PDF for Testing

For testing your implementation, download and use this technical paper:

**📄 Test Document**: ["Attention Is All You Need" Paper](https://arxiv.org/pdf/1706.03762.pdf)
- **Direct Download**: `https://arxiv.org/pdf/1706.03762.pdf`
- **File Name**: `1706.03762v7.pdf` (save as this name)
- **Paper Title**: "Attention Is All You Need" (Transformer Architecture)

### Why This PDF is Perfect for Testing

This technical paper contains all the elements needed to test your system:

✅ **Text Content**: 
- Multiple pages with academic text
- Abstract, introduction, methodology sections
- References and citations

✅ **Images/Figures**: 
- Architecture diagrams (Figure 1: Transformer model architecture)
- Attention visualization diagrams
- Model comparison charts

✅ **Tables**: 
- Performance comparison tables
- Experimental results
- Hyperparameter settings

✅ **Complex Layout**: 
- Two-column academic format
- Mathematical equations
- Mixed content types

### Download Instructions

```bash
# Download the test PDF
wget https://arxiv.org/pdf/1706.03762.pdf -O 1706.03762v7.pdf

# Or use curl
curl -o 1706.03762v7.pdf https://arxiv.org/pdf/1706.03762.pdf
```

**Important**: Use this specific PDF for your testing and demonstrations to ensure consistent evaluation across all submissions.

---

## Implementation Guidelines

Refer to the service skeleton files for detailed implementation guidance:
- `backend/app/services/document_processor.py` - Document processing guidelines
- `backend/app/services/vector_store.py` - Vector store implementation tips
- `backend/app/services/chat_engine.py` - Chat engine implementation tips

Each file contains detailed TODO comments with implementation hints and examples.

---

## Troubleshooting

### Document Processing Issues
**Problem**: Docling can't extract tables
**Solution**: 
- Check PDF format (ensure it's not scanned image)
- Add fallback parsing logic
- Manually define table structure patterns

### LLM API Costs
**Problem**: OpenAI API is expensive
**Solution**: Use free alternatives
- Use caching for repeated queries
- Use cheaper models (gpt-3.5-turbo)
- Use local LLM (Ollama) for development

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

## Free LLM Options

You don't need to pay for OpenAI API! Here are free alternatives:

### Option 1: Ollama (Recommended for Development)

**Completely free, runs locally on your machine**

1. **Install Ollama**
```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

2. **Download a model**
```bash
# Llama 3.2 (3B - fast, good for development)
ollama pull llama3.2

# Or Llama 3.1 (8B - better quality)
ollama pull llama3.1

# Or Mistral (7B - good balance)
ollama pull mistral
```

3. **Update your .env**
```bash
# Use Ollama instead of OpenAI
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Pros**: Free, private, no API limits, works offline
**Cons**: Requires decent hardware (8GB+ RAM), slower than cloud APIs

---

### Option 2: Google Gemini (Free Tier)

**Free tier: 60 requests per minute**

1. **Get free API key**
   - Go to https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy your key

2. **Update .env**
```bash
GOOGLE_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini
```

**Pros**: Free, fast, good quality
**Cons**: Rate limits, requires internet

---

### Option 3: Groq (Free Tier)

**Free tier: Very fast inference, generous limits**

1. **Get free API key**
   - Go to https://console.groq.com
   - Sign up and get API key

2. **Update .env**
```bash
GROQ_API_KEY=your-groq-api-key
LLM_PROVIDER=groq
```

**Pros**: Free, extremely fast, good quality
**Cons**: Rate limits, requires internet

---

### Comparison Table

| Provider | Cost | Speed | Quality | Setup |
|----------|------|-------|---------|-------|
| **Ollama** | Free | Medium | Good | Easy |
| **Gemini** | Free | Fast | Very Good | Very Easy |
| **Groq** | Free | Very Fast | Good | Very Easy |
| OpenAI | Paid | Fast | Excellent | Very Easy |

**Recommended**: Use **Ollama** for development (free, no limits)

---

## FAQ

**Q: Docling won't install.**
A: Try `pip install docling` or use the Docker image.

**Q: I don't have an OpenAI API key.**
A: You can install Ollama locally and use a free LLM (see Free LLM Options section).

**Q: Where should I save images?**
A: Save to `backend/uploads/images/` directory and store only the path in DB.

**Q: How should I display tables?**
A: Render tables as images or display JSON data as HTML tables in frontend.

**Q: Where do I get the test PDF file?**
A: Download from `https://arxiv.org/pdf/1706.03762.pdf` and save as `1706.03762v7.pdf`. This is the "Attention Is All You Need" paper.

**Q: Can I use a different PDF for testing?**
A: For development, yes. But your final submission must demonstrate working examples with the provided "Attention Is All You Need" paper for consistent evaluation.

**Q: What if my system can't process the test PDF?**
A: The test PDF is a standard academic paper. If your system can't handle it, there may be issues with your PDF parsing or Docling integration that need to be fixed.

**Q: How do I test the system locally?**
A: Follow the Getting Started section, download the test PDF, upload it through your frontend, and try the sample questions from the Test Scenarios section.

---

## Questions?

If you have any questions, please create an issue or contact us via email.

Good luck!

---

## Tips for Success

1. **Start Simple**: Get core features working before adding advanced features
2. **Test Early**: Test document processing with sample PDF immediately
3. **Use Tools**: Leverage Docling, LangChain to save time
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

**Version**: 1.0  
**Last Updated**: 2025-11-03  
**Author**: InterOpera-Apps Hiring Team
