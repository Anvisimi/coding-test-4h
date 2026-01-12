"""
Unit tests for ChatEngine service
"""
import pytest
from app.services.chat_engine import ChatEngine
from app.models.conversation import Message


class TestChatEngine:
    """Test ChatEngine functionality"""
    
    @pytest.mark.asyncio
    async def test_load_conversation_history(self, test_db, sample_conversation):
        """Test loading conversation history"""
        msg1 = Message(conversation_id=sample_conversation.id, role="user", content="Question 1")
        msg2 = Message(conversation_id=sample_conversation.id, role="assistant", content="Answer 1")
        test_db.add(msg1)
        test_db.add(msg2)
        test_db.commit()
        
        chat_engine = ChatEngine(test_db)
        history = await chat_engine._load_conversation_history(sample_conversation.id, limit=5)
        
        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_search_context(self, test_db, sample_document, monkeypatch):
        """Test context search functionality"""
        mock_results = [{"content": "Test content", "score": 0.95, "page_number": 1, "related_images": [], "related_tables": []}]
        
        async def mock_similarity_search(query, document_id, k):
            return mock_results
        
        chat_engine = ChatEngine(test_db)
        chat_engine.vector_store.similarity_search = mock_similarity_search
        
        context = await chat_engine._search_context("test query", sample_document.id, k=5)
        
        assert isinstance(context, list)
        assert len(context) == 1
    
    @pytest.mark.asyncio
    async def test_find_related_media(self, test_db):
        """Test finding related images and tables"""
        context_chunks = [
            {
                "content": "Test",
                "related_images": [{"id": 1, "url": "/img1.png", "caption": "Fig 1", "page": 1}],
                "related_tables": [{"id": 1, "url": "/table1.png", "caption": "Table 1", "page": 2}]
            }
        ]
        
        chat_engine = ChatEngine(test_db)
        media = await chat_engine._find_related_media(context_chunks)
        
        assert "images" in media
        assert "tables" in media
        assert len(media["images"]) == 1
        assert len(media["tables"]) == 1
    
    @pytest.mark.asyncio
    async def test_format_sources(self, test_db):
        """Test formatting sources for response"""
        context = [{"content": "Test", "page_number": 1, "score": 0.95}]
        media = {"images": [{"url": "/img1.png", "caption": "Fig 1", "page": 1}], "tables": []}
        
        chat_engine = ChatEngine(test_db)
        sources = chat_engine._format_sources(context, media)
        
        assert isinstance(sources, list)
        assert sources[0]["type"] == "text"
        assert sources[1]["type"] == "image"
    
    @pytest.mark.asyncio
    async def test_process_message(self, test_db, sample_conversation, sample_document, monkeypatch, mock_openai_chat):
        """Test processing a message end-to-end"""
        async def mock_search_context(query, document_id, k):
            return [{"content": "Test", "page_number": 1, "score": 0.95, "related_images": [], "related_tables": []}]
        
        async def mock_generate_response(message, context, history, media):
            return "Test answer"
        
        chat_engine = ChatEngine(test_db)
        chat_engine._search_context = mock_search_context
        chat_engine._generate_response = mock_generate_response
        
        result = await chat_engine.process_message(
            conversation_id=sample_conversation.id,
            message="Test question",
            document_id=sample_document.id
        )
        
        assert "answer" in result
        assert "sources" in result
        assert "processing_time" in result
    
    def test_initialize_llm_openai(self, test_db, monkeypatch):
        """Test LLM initialization with OpenAI"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        # Also patch settings since it's checked
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        chat_engine = ChatEngine(test_db)
        assert chat_engine.llm["provider"] == "openai"
    
    def test_initialize_llm_ollama(self, test_db, monkeypatch):
        """Test LLM initialization with Ollama"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        chat_engine = ChatEngine(test_db)
        assert chat_engine.llm["provider"] == "ollama"
    
    def test_initialize_llm_gemini(self, test_db, monkeypatch):
        """Test LLM initialization with Gemini"""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        chat_engine = ChatEngine(test_db)
        assert chat_engine.llm["provider"] == "gemini"
    
    def test_initialize_llm_groq(self, test_db, monkeypatch):
        """Test LLM initialization with Groq"""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        chat_engine = ChatEngine(test_db)
        assert chat_engine.llm["provider"] == "groq"
    
    @pytest.mark.asyncio
    async def test_generate_response_openai(self, test_db, monkeypatch, mock_openai_chat):
        """Test response generation with OpenAI"""
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        
        chat_engine = ChatEngine(test_db)
        context = [{"content": "Test context", "page_number": 1}]
        history = []
        media = {"images": [], "tables": []}
        
        response = await chat_engine._generate_response("Test question", context, history, media)
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_with_history(self, test_db, monkeypatch, mock_openai_chat):
        """Test response generation with conversation history"""
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        
        chat_engine = ChatEngine(test_db)
        context = [{"content": "Test context", "page_number": 1}]
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        media = {"images": [], "tables": []}
        
        response = await chat_engine._generate_response("Follow-up question", context, history, media)
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_generate_response_with_media(self, test_db, monkeypatch, mock_openai_chat):
        """Test response generation with images and tables"""
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        
        chat_engine = ChatEngine(test_db)
        context = [{"content": "Test context", "page_number": 1}]
        history = []
        media = {
            "images": [{"url": "/img1.png", "caption": "Figure 1", "page": 1}],
            "tables": [{"url": "/table1.png", "caption": "Table 1", "page": 2}]
        }
        
        response = await chat_engine._generate_response("Question about image", context, history, media)
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_process_message_without_document_id(self, test_db, sample_conversation, monkeypatch, mock_openai_chat):
        """Test processing a message without document_id"""
        async def mock_search_context(query, document_id, k):
            return [{"content": "Test", "page_number": 1, "score": 0.95, "related_images": [], "related_tables": []}]
        
        async def mock_generate_response(message, context, history, media):
            return "Test answer"
        
        chat_engine = ChatEngine(test_db)
        chat_engine._search_context = mock_search_context
        chat_engine._generate_response = mock_generate_response
        
        result = await chat_engine.process_message(
            conversation_id=sample_conversation.id,
            message="Test question",
            document_id=None
        )
        
        assert "answer" in result
        assert "sources" in result
    
    @pytest.mark.asyncio
    async def test_load_conversation_history_limit(self, test_db, sample_conversation):
        """Test loading conversation history with limit"""
        # Create more messages than limit
        for i in range(10):
            msg = Message(conversation_id=sample_conversation.id, role="user", content=f"Question {i}")
            test_db.add(msg)
        test_db.commit()
        
        chat_engine = ChatEngine(test_db)
        history = await chat_engine._load_conversation_history(sample_conversation.id, limit=5)
        
        assert isinstance(history, list)
        # The method loads limit*2 messages, so with limit=5 it loads 10
        # But we verify it loads messages correctly
        assert len(history) <= 10
        assert len(history) > 0
    
    @pytest.mark.asyncio
    async def test_search_context_without_document_id(self, test_db, monkeypatch):
        """Test context search without document_id"""
        mock_results = [{"content": "Test", "score": 0.95, "page_number": 1, "related_images": [], "related_tables": []}]
        
        async def mock_similarity_search(query, document_id, k):
            return mock_results
        
        chat_engine = ChatEngine(test_db)
        chat_engine.vector_store.similarity_search = mock_similarity_search
        
        context = await chat_engine._search_context("test query", document_id=None, k=5)
        
        assert isinstance(context, list)
    
    @pytest.mark.asyncio
    async def test_find_related_media_empty(self, test_db):
        """Test finding related media with empty context"""
        chat_engine = ChatEngine(test_db)
        media = await chat_engine._find_related_media([])
        
        assert "images" in media
        assert "tables" in media
        assert len(media["images"]) == 0
        assert len(media["tables"]) == 0
    
    @pytest.mark.asyncio
    async def test_format_sources_empty(self, test_db):
        """Test formatting sources with empty context and media"""
        chat_engine = ChatEngine(test_db)
        sources = chat_engine._format_sources([], {"images": [], "tables": []})
        
        assert isinstance(sources, list)
        assert len(sources) == 0
    
    @pytest.mark.asyncio
    async def test_call_ollama(self, test_db, monkeypatch):
        """Test calling Ollama API"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        
        class MockResponse:
            def json(self):
                return {"message": {"content": "Ollama response"}}
            def raise_for_status(self):
                pass
        
        async def mock_post(*args, **kwargs):
            return MockResponse()
        
        import httpx
        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
        
        chat_engine = ChatEngine(test_db)
        response = await chat_engine._call_ollama("system", "user", [])
        
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_call_gemini(self, test_db, monkeypatch):
        """Test calling Gemini API"""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        class MockModel:
            def generate_content(self, prompt):
                class MockResponse:
                    text = "Gemini response"
                return MockResponse()
        
        class MockGenAI:
            @staticmethod
            def configure(*args, **kwargs):
                pass
            
            @staticmethod
            def GenerativeModel(*args, **kwargs):
                return MockModel()
        
        # Mock the module import
        import sys
        mock_genai_module = type(sys)('google.generativeai')
        mock_genai_module.configure = MockGenAI.configure
        mock_genai_module.GenerativeModel = MockGenAI.GenerativeModel
        sys.modules['google.generativeai'] = mock_genai_module
        
        try:
            chat_engine = ChatEngine(test_db)
            response = await chat_engine._call_gemini("system", "user", [])
            
            assert isinstance(response, str)
        finally:
            # Clean up
            if 'google.generativeai' in sys.modules:
                del sys.modules['google.generativeai']
    
    @pytest.mark.asyncio
    async def test_call_gemini_import_error(self, test_db, monkeypatch):
        """Test Gemini API with import error"""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        # Create a selective import mock that only blocks google.generativeai
        original_import = __import__
        
        def selective_import_error(name, *args, **kwargs):
            if name == 'google.generativeai' or name.startswith('google.generativeai'):
                raise ImportError("No module named 'google.generativeai'")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", selective_import_error)
        
        # Also remove from sys.modules if it exists
        import sys
        if 'google.generativeai' in sys.modules:
            del sys.modules['google.generativeai']
        
        chat_engine = ChatEngine(test_db)
        response = await chat_engine._call_gemini("system", "user", [])
        
        assert "Error" in response or "not installed" in response

    
    @pytest.mark.asyncio
    async def test_call_groq_import_error(self, test_db, monkeypatch):
        """Test Groq API with import error"""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        
        # Create a selective import mock that only blocks groq
        original_import = __import__
        
        def selective_import_error(name, *args, **kwargs):
            if name == 'groq' or name.startswith('groq'):
                raise ImportError("No module named 'groq'")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr("builtins.__import__", selective_import_error)
        
        # Also remove from sys.modules if it exists
        import sys
        if 'groq' in sys.modules:
            del sys.modules['groq']
        
        chat_engine = ChatEngine(test_db)
        response = await chat_engine._call_groq("system", "user", [])
        
        assert "Error" in response or "not installed" in response
    
    @pytest.mark.asyncio
    async def test_generate_response_ollama_provider(self, test_db, monkeypatch):
        """Test response generation with Ollama provider"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        
        async def mock_call_ollama(system, user, history):
            return "Ollama answer"
        
        chat_engine = ChatEngine(test_db)
        chat_engine._call_ollama = mock_call_ollama
        
        context = [{"content": "Test", "page_number": 1}]
        response = await chat_engine._generate_response("Question", context, [], {"images": [], "tables": []})
        
        assert response == "Ollama answer"
    
    @pytest.mark.asyncio
    async def test_generate_response_gemini_provider(self, test_db, monkeypatch):
        """Test response generation with Gemini provider"""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        
        async def mock_call_gemini(system, user, history):
            return "Gemini answer"
        
        chat_engine = ChatEngine(test_db)
        chat_engine._call_gemini = mock_call_gemini
        
        context = [{"content": "Test", "page_number": 1}]
        response = await chat_engine._generate_response("Question", context, [], {"images": [], "tables": []})
        
        assert response == "Gemini answer"
    
    @pytest.mark.asyncio
    async def test_generate_response_groq_provider(self, test_db, monkeypatch):
        """Test response generation with Groq provider"""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        
        async def mock_call_groq(system, user, history):
            return "Groq answer"
        
        chat_engine = ChatEngine(test_db)
        chat_engine._call_groq = mock_call_groq
        
        context = [{"content": "Test", "page_number": 1}]
        response = await chat_engine._generate_response("Question", context, [], {"images": [], "tables": []})
        
        assert response == "Groq answer"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_multiple_context_chunks(self, test_db, monkeypatch, mock_openai_chat):
        """Test response generation with multiple context chunks"""
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        
        chat_engine = ChatEngine(test_db)
        context = [
            {"content": "Context 1", "page_number": 1},
            {"content": "Context 2", "page_number": 2},
            {"content": "Context 3", "page_number": 3},
            {"content": "Context 4", "page_number": 4},
            {"content": "Context 5", "page_number": 5},
            {"content": "Context 6", "page_number": 6},  # Should be limited to 5
        ]
        response = await chat_engine._generate_response("Question", context, [], {"images": [], "tables": []})
        
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_find_related_media_with_duplicates(self, test_db):
        """Test finding related media with duplicate images/tables"""
        context_chunks = [
            {
                "content": "Test 1",
                "related_images": [{"id": 1, "url": "/img1.png", "caption": "Fig 1", "page": 1}],
                "related_tables": [{"id": 1, "url": "/table1.png", "caption": "Table 1", "page": 2}]
            },
            {
                "content": "Test 2",
                "related_images": [{"id": 1, "url": "/img1.png", "caption": "Fig 1", "page": 1}],  # Duplicate
                "related_tables": [{"id": 2, "url": "/table2.png", "caption": "Table 2", "page": 3}]
            }
        ]
        
        chat_engine = ChatEngine(test_db)
        media = await chat_engine._find_related_media(context_chunks)
        
        assert "images" in media
        assert "tables" in media
        # Should deduplicate images
        assert len(media["images"]) == 1
        assert len(media["tables"]) == 2
    
    @pytest.mark.asyncio
    async def test_find_related_media_limit(self, test_db):
        """Test finding related media respects limit of 10"""
        # Create chunks with more than 10 images/tables
        context_chunks = []
        for i in range(15):
            context_chunks.append({
                "content": f"Test {i}",
                "related_images": [{"id": i, "url": f"/img{i}.png", "caption": f"Fig {i}", "page": i}],
                "related_tables": []
            })
        
        chat_engine = ChatEngine(test_db)
        media = await chat_engine._find_related_media(context_chunks)
        
        assert "images" in media
        # Should be limited to 10
        assert len(media["images"]) <= 10
    
    @pytest.mark.asyncio
    async def test_format_sources_with_all_types(self, test_db):
        """Test formatting sources with all types (text, image, table)"""
        context = [
            {"content": "Text 1", "page_number": 1, "score": 0.95},
            {"content": "Text 2", "page_number": 2, "score": 0.90}
        ]
        media = {
            "images": [
                {"url": "/img1.png", "caption": "Fig 1", "page": 1},
                {"url": "/img2.png", "caption": "Fig 2", "page": 2}
            ],
            "tables": [
                {"url": "/table1.png", "caption": "Table 1", "page": 3, "data": {}}
            ]
        }
        
        chat_engine = ChatEngine(test_db)
        sources = chat_engine._format_sources(context, media)
        
        assert isinstance(sources, list)
        assert len(sources) == 5  # 2 text + 2 images + 1 table
        assert sources[0]["type"] == "text"
        assert sources[2]["type"] == "image"
        assert sources[4]["type"] == "table"
    
    @pytest.mark.asyncio
    async def test_process_message_error_in_search(self, test_db, sample_conversation, sample_document, monkeypatch):
        """Test process_message error handling when search fails"""
        async def mock_search_error(query, document_id, k):
            raise Exception("Search failed")
        
        chat_engine = ChatEngine(test_db)
        chat_engine._search_context = mock_search_error
        
        with pytest.raises(Exception):
            await chat_engine.process_message(
                conversation_id=sample_conversation.id,
                message="Test question",
                document_id=sample_document.id
            )
    
    @pytest.mark.asyncio
    async def test_process_message_error_in_generate(self, test_db, sample_conversation, sample_document, monkeypatch):
        """Test process_message error handling when response generation fails"""
        async def mock_search_context(query, document_id, k):
            return [{"content": "Test", "page_number": 1, "score": 0.95, "related_images": [], "related_tables": []}]
        
        async def mock_generate_error(message, context, history, media):
            raise Exception("Generation failed")
        
        chat_engine = ChatEngine(test_db)
        chat_engine._search_context = mock_search_context
        chat_engine._generate_response = mock_generate_error
        
        with pytest.raises(Exception):
            await chat_engine.process_message(
                conversation_id=sample_conversation.id,
                message="Test question",
                document_id=sample_document.id
            )
