"""
Chat engine service for multimodal RAG.

TODO: Implement this service to:
1. Process user messages
2. Search for relevant context using vector store
3. Find related images and tables
4. Generate responses using LLM
5. Support multi-turn conversations
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation, Message
from app.services.vector_store import VectorStore
from app.core.config import settings
import time
import os


class ChatEngine:
    """
    Multimodal chat engine with RAG.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """
        Initialize LLM based on configuration.
        """
        # Check for LLM provider setting
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if llm_provider == "ollama":
            # Ollama setup
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
            self.llm = {"provider": "ollama", "base_url": ollama_base_url, "model": ollama_model}
        elif llm_provider == "gemini":
            # Google Gemini
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.llm = {"provider": "gemini", "api_key": api_key}
            else:
                self.llm = {"provider": "openai"}  # Fallback
        elif llm_provider == "groq":
            # Groq
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                self.llm = {"provider": "groq", "api_key": groq_api_key}
            else:
                self.llm = {"provider": "openai"}  # Fallback
        else:
            # Default to OpenAI
            if settings.OPENAI_API_KEY:
                self.llm = {"provider": "openai"}
            else:
                # Try to use Ollama as fallback
                self.llm = {"provider": "ollama", "base_url": "http://localhost:11434", "model": "llama3.2"}
    
    async def process_message(
        self,
        conversation_id: int,
        message: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate multimodal response.
        
        Implementation steps:
        1. Load conversation history (for multi-turn support)
        2. Search vector store for relevant context
        3. Find related images and tables
        4. Build prompt with context and history
        5. Generate response using LLM
        6. Format response with sources (text, images, tables)
        
        Args:
            conversation_id: Conversation ID
            message: User message
            document_id: Optional document ID to scope search
            
        Returns:
            {
                "answer": "...",
                "sources": [
                    {
                        "type": "text",
                        "content": "...",
                        "page": 3,
                        "score": 0.95
                    },
                    {
                        "type": "image",
                        "url": "/uploads/images/xxx.png",
                        "caption": "Figure 1: ...",
                        "page": 3
                    },
                    {
                        "type": "table",
                        "url": "/uploads/tables/yyy.png",
                        "caption": "Table 1: ...",
                        "page": 5,
                        "data": {...}  # structured table data
                    }
                ],
                "processing_time": 2.5
            }
        """
        start_time = time.time()
        
        try:
            # Load conversation history
            history = await self._load_conversation_history(conversation_id)
            
            # Search for relevant context
            context = await self._search_context(message, document_id, k=settings.TOP_K_RESULTS)
            
            # Find related media
            media = await self._find_related_media(context)
            
            # Generate response
            answer = await self._generate_response(message, context, history, media)
            
            # Format sources
            sources = self._format_sources(context, media)
            
            # Save user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )
            self.db.add(user_msg)
            self.db.commit()
            
            # Save assistant response
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources
            )
            self.db.add(assistant_msg)
            self.db.commit()
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "sources": sources,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            raise
    
    async def _load_conversation_history(
        self,
        conversation_id: int,
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Load recent conversation history.
        
        TODO: Implement conversation history loading
        - Load last N messages from conversation
        - Format for LLM context
        - Include both user and assistant messages
        
        Returns:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at.desc()).limit(limit * 2).all()
        
        # Reverse to get chronological order
        messages.reverse()
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    async def _search_context(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using vector store.
        
        TODO: Implement context search
        - Use vector store similarity search
        - Filter by document if specified
        - Return relevant chunks with metadata
        """
        return await self.vector_store.similarity_search(query, document_id, k)
    
    async def _find_related_media(
        self,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find related images and tables from context chunks.
        
        TODO: Implement related media finding
        - Extract image/table references from chunk metadata
        - Query database for actual image/table records
        - Return with URLs for frontend display
        
        Returns:
            {
                "images": [
                    {
                        "url": "/uploads/images/xxx.png",
                        "caption": "...",
                        "page": 3
                    }
                ],
                "tables": [
                    {
                        "url": "/uploads/tables/yyy.png",
                        "caption": "...",
                        "page": 5,
                        "data": {...}
                    }
                ]
            }
        """
        # Extract unique images and tables from context chunks
        all_images = []
        all_tables = []
        seen_image_ids = set()
        seen_table_ids = set()
        
        for chunk in context_chunks:
            # Add images from chunk
            for img in chunk.get("related_images", []):
                if img.get("id") and img["id"] not in seen_image_ids:
                    all_images.append(img)
                    seen_image_ids.add(img["id"])
            
            # Add tables from chunk
            for tbl in chunk.get("related_tables", []):
                if tbl.get("id") and tbl["id"] not in seen_table_ids:
                    all_tables.append(tbl)
                    seen_table_ids.add(tbl["id"])
        
        return {
            "images": all_images[:10],  # Limit to top 10
            "tables": all_tables[:10]   # Limit to top 10
        }
    
    async def _generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Generate response using LLM.
        
        TODO: Implement LLM response generation
        - Build comprehensive prompt with:
          - System instructions
          - Conversation history
          - Retrieved context
          - Available images/tables
        - Call LLM API
        - Return generated answer
        
        Prompt engineering tips:
        - Instruct LLM to reference images/tables when relevant
        - Include context from previous messages
        - Ask LLM to cite sources
        - Format for good UX (bullet points, etc.)
        """
        # Build context text
        context_text = ""
        for i, chunk in enumerate(context[:5], 1):  # Top 5 chunks
            context_text += f"\n[Context {i} - Page {chunk.get('page_number', '?')}]:\n{chunk['content']}\n"
        
        # Build media references
        media_refs = ""
        if media.get("images"):
            media_refs += "\n\nAvailable Images:\n"
            for i, img in enumerate(media["images"][:5], 1):
                caption = img.get("caption", "Image")
                page = img.get("page", "?")
                media_refs += f"- Image {i} (Page {page}): {caption}\n"
        
        if media.get("tables"):
            media_refs += "\n\nAvailable Tables:\n"
            for i, tbl in enumerate(media["tables"][:5], 1):
                caption = tbl.get("caption", "Table")
                page = tbl.get("page", "?")
                media_refs += f"- Table {i} (Page {page}): {caption}\n"
        
        # Build system prompt
        system_prompt = """You are a helpful assistant that answers questions based on provided document context.
When answering:
- Use the provided context to answer the question accurately
- Reference specific pages when mentioning information
- If images or tables are available and relevant, mention them in your response
- If you don't know the answer based on the context, say so
- Format your response clearly with bullet points or paragraphs as appropriate
- Cite page numbers when referencing specific information"""
        
        # Build user prompt
        user_prompt = f"""Based on the following document context, please answer the user's question.

Document Context:
{context_text}
{media_refs}

User Question: {message}

Please provide a comprehensive answer based on the context above."""
        
        # Call LLM based on provider
        provider = self.llm.get("provider", "openai")
        
        if provider == "ollama":
            return await self._call_ollama(system_prompt, user_prompt, history)
        elif provider == "gemini":
            return await self._call_gemini(system_prompt, user_prompt, history)
        elif provider == "groq":
            return await self._call_groq(system_prompt, user_prompt, history)
        else:
            return await self._call_openai(system_prompt, user_prompt, history)
    
    async def _call_openai(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]]) -> str:
        """Call OpenAI API"""
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history (last few messages)
        for h in history[-4:]:  # Last 4 messages for context
            messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def _call_ollama(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]]) -> str:
        """Call Ollama API"""
        import httpx
        
        base_url = self.llm.get("base_url", "http://localhost:11434")
        model = self.llm.get("model", "llama3.2")
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history
        for h in history[-4:]:
            messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
    
    async def _call_gemini(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]]) -> str:
        """Call Google Gemini API"""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return "Error: google-generativeai package not installed. Please install it with: pip install google-generativeai"
            
            genai.configure(api_key=self.llm.get("api_key"))
            model = genai.GenerativeModel('gemini-pro')
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback to a simple response
            return "I apologize, but I'm having trouble generating a response. Please try again."
    
    async def _call_groq(self, system_prompt: str, user_prompt: str, history: List[Dict[str, str]]) -> str:
        """Call Groq API"""
        try:
            try:
                from groq import Groq
            except ImportError:
                return "Error: groq package not installed. Please install it with: pip install groq"
            
            client = Groq(api_key=self.llm.get("api_key"))
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add history
            for h in history[-4:]:
                messages.append({"role": h["role"], "content": h["content"]})
            
            messages.append({"role": "user", "content": user_prompt})
            
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Groq's fast model
                messages=messages
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return "I apologize, but I'm having trouble generating a response. Please try again."
    
    def _format_sources(
        self,
        context: List[Dict[str, Any]],
        media: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Format sources for response.
        
        This is implemented as an example.
        """
        sources = []
        
        # Add text sources
        for chunk in context[:3]:  # Top 3 text chunks
            sources.append({
                "type": "text",
                "content": chunk["content"],
                "page": chunk.get("page_number"),
                "score": chunk.get("score", 0.0)
            })
        
        # Add image sources
        for image in media.get("images", []):
            sources.append({
                "type": "image",
                "url": image["url"],
                "caption": image.get("caption"),
                "page": image.get("page")
            })
        
        # Add table sources
        for table in media.get("tables", []):
            sources.append({
                "type": "table",
                "url": table["url"],
                "caption": table.get("caption"),
                "page": table.get("page"),
                "data": table.get("data")
            })
        
        return sources
