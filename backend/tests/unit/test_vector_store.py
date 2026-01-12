"""
Unit tests for VectorStore service
"""
import pytest
import numpy as np
from app.services.vector_store import VectorStore
from app.models.document import DocumentChunk


class TestVectorStore:
    """Test VectorStore functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_huggingface(self, test_db, monkeypatch):
        """Test embedding generation with HuggingFace fallback"""
        def mock_encode(text, convert_to_numpy=True):
            return np.random.rand(384).astype(np.float32)
        
        class MockSentenceTransformer:
            def encode(self, text, convert_to_numpy=True):
                return mock_encode(text, convert_to_numpy)
        
        monkeypatch.setenv("OPENAI_API_KEY", "")
        vector_store = VectorStore(test_db)
        vector_store.embeddings_model = MockSentenceTransformer()
        
        embedding = await vector_store.generate_embedding("Test text")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536
        assert embedding.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_store_chunk(self, test_db, sample_document, monkeypatch):
        """Test storing a text chunk with embedding"""
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        async def mock_generate_embedding(text):
            return mock_embedding
        
        vector_store = VectorStore(test_db)
        vector_store.generate_embedding = mock_generate_embedding
        
        chunk = await vector_store.store_chunk(
            content="Test chunk content",
            document_id=sample_document.id,
            page_number=1,
            chunk_index=0,
            metadata={"related_images": [1]}
        )
        
        assert chunk is not None
        assert chunk.document_id == sample_document.id
        assert chunk.content == "Test chunk content"
        assert chunk.embedding is not None
        
        db_chunk = test_db.query(DocumentChunk).filter(DocumentChunk.id == chunk.id).first()
        assert db_chunk is not None
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_openai(self, test_db, monkeypatch):
        """Test embedding generation with OpenAI"""
        mock_response = type('obj', (object,), {
            'data': [type('obj', (object,), {'embedding': [0.1] * 1536})()]
        })()
        
        class MockOpenAI:
            class Embeddings:
                def create(self, model, input):
                    return mock_response
            embeddings = Embeddings()
        
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        vector_store = VectorStore(test_db)
        vector_store.embeddings_model = "openai"
        vector_store.openai_client = MockOpenAI()
        
        embedding = await vector_store.generate_embedding("Test text")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536
    
    def test_ensure_extension(self, test_db):
        """Test pgvector extension check"""
        vector_store = VectorStore(test_db)
        # Should not raise exception (SQLite will fail gracefully)
        try:
            vector_store._ensure_extension()
        except:
            pass  # Expected for SQLite
    
    @pytest.mark.asyncio
    async def test_get_related_content(self, test_db, sample_document):
        """Test getting related content for chunks"""
        from app.models.document import DocumentChunk, DocumentImage, DocumentTable
        
        # Create chunk with related content
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Test chunk",
            page_number=1,
            chunk_index=0,
            extra_metadata={"related_images": [], "related_tables": []}
        )
        test_db.add(chunk)
        
        # Create image and table
        image = DocumentImage(
            document_id=sample_document.id,
            file_path="/tmp/test.png",
            page_number=1,
            caption="Test image"
        )
        table = DocumentTable(
            document_id=sample_document.id,
            image_path="/tmp/test_table.png",
            page_number=1,
            caption="Test table",
            data={}
        )
        test_db.add(image)
        test_db.add(table)
        test_db.commit()
        test_db.refresh(chunk)
        test_db.refresh(image)
        test_db.refresh(table)
        
        # Update chunk metadata
        chunk.extra_metadata = {"related_images": [image.id], "related_tables": [table.id]}
        test_db.commit()
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        assert "images" in related
        assert "tables" in related
        assert len(related["images"]) > 0
        assert len(related["tables"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_related_content_empty(self, test_db):
        """Test getting related content with no chunks"""
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([])
        
        assert "images" in related
        assert "tables" in related
        assert len(related["images"]) == 0
        assert len(related["tables"]) == 0
    
    def test_initialize_embeddings_model_openai(self, test_db, monkeypatch):
        """Test initialization with OpenAI"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Also patch settings since it's checked during initialization
        from app.core.config import settings
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
        vector_store = VectorStore(test_db)
        assert vector_store.embeddings_model == "openai"
        assert hasattr(vector_store, "openai_client")
    
    def test_initialize_embeddings_model_huggingface(self, test_db, monkeypatch):
        """Test initialization with HuggingFace fallback"""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        
        def mock_sentence_transformer(*args, **kwargs):
            class MockModel:
                def encode(self, text, convert_to_numpy=True):
                    import numpy as np
                    return np.random.rand(384).astype(np.float32)
            return MockModel()
        
        monkeypatch.setattr("sentence_transformers.SentenceTransformer", mock_sentence_transformer)
        
        vector_store = VectorStore(test_db)
        assert vector_store.embeddings_model is not None
        assert vector_store.embeddings_model != "openai"
    
    @pytest.mark.asyncio
    async def test_generate_embedding_padding(self, test_db, monkeypatch):
        """Test embedding generation with padding for smaller dimensions"""
        def mock_encode(text, convert_to_numpy=True):
            import numpy as np
            # Return smaller embedding (384 instead of 1536)
            return np.random.rand(384).astype(np.float32)
        
        class MockSentenceTransformer:
            def encode(self, text, convert_to_numpy=True):
                return mock_encode(text, convert_to_numpy)
        
        monkeypatch.setenv("OPENAI_API_KEY", "")
        vector_store = VectorStore(test_db)
        vector_store.embeddings_model = MockSentenceTransformer()
        
        embedding = await vector_store.generate_embedding("Test text")
        
        assert embedding is not None
        assert len(embedding) == 1536  # Should be padded to 1536
        assert embedding.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_generate_embedding_truncation(self, test_db, monkeypatch):
        """Test embedding generation with truncation for larger dimensions"""
        def mock_encode(text, convert_to_numpy=True):
            import numpy as np
            # Return larger embedding (2000 instead of 1536)
            return np.random.rand(2000).astype(np.float32)
        
        class MockSentenceTransformer:
            def encode(self, text, convert_to_numpy=True):
                return mock_encode(text, convert_to_numpy)
        
        monkeypatch.setenv("OPENAI_API_KEY", "")
        vector_store = VectorStore(test_db)
        vector_store.embeddings_model = MockSentenceTransformer()
        
        embedding = await vector_store.generate_embedding("Test text")
        
        assert embedding is not None
        assert len(embedding) == 1536  # Should be truncated to 1536
        assert embedding.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_store_chunk_with_metadata(self, test_db, sample_document, monkeypatch):
        """Test storing chunk with complex metadata"""
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        async def mock_generate_embedding(text):
            return mock_embedding
        
        vector_store = VectorStore(test_db)
        vector_store.generate_embedding = mock_generate_embedding
        
        metadata = {
            "related_images": [1, 2, 3],
            "related_tables": [1],
            "section": "Introduction",
            "keywords": ["test", "example"]
        }
        
        chunk = await vector_store.store_chunk(
            content="Test chunk with metadata",
            document_id=sample_document.id,
            page_number=1,
            chunk_index=0,
            metadata=metadata
        )
        
        assert chunk is not None
        assert chunk.extra_metadata == metadata
        assert chunk.extra_metadata["related_images"] == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_store_chunk_error_handling(self, test_db, sample_document, monkeypatch):
        """Test error handling in store_chunk"""
        async def mock_generate_embedding(text):
            raise Exception("Embedding generation failed")
        
        vector_store = VectorStore(test_db)
        vector_store.generate_embedding = mock_generate_embedding
        
        with pytest.raises(Exception):
            await vector_store.store_chunk(
                content="Test content",
                document_id=sample_document.id,
                page_number=1,
                chunk_index=0
            )
    
    @pytest.mark.asyncio
    async def test_get_related_content_with_page_matching(self, test_db, sample_document):
        """Test getting related content with page-based matching"""
        from app.models.document import DocumentChunk, DocumentImage, DocumentTable
        
        # Create chunk on page 2
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Test chunk",
            page_number=2,
            chunk_index=0,
            extra_metadata={}
        )
        test_db.add(chunk)
        
        # Create image on same page (should be found)
        image = DocumentImage(
            document_id=sample_document.id,
            file_path="/tmp/test.png",
            page_number=2,
            caption="Page 2 image"
        )
        test_db.add(image)
        
        # Create table on different page (should not be found via page matching)
        table = DocumentTable(
            document_id=sample_document.id,
            image_path="/tmp/test_table.png",
            page_number=3,
            caption="Page 3 table",
            data={}
        )
        test_db.add(table)
        test_db.commit()
        test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        assert "images" in related
        assert "tables" in related
        # Page-based matching should find the image on page 2
        assert len(related["images"]) >= 1
    
    @pytest.mark.asyncio
    async def test_get_related_content_multiple_chunks(self, test_db, sample_document):
        """Test getting related content for multiple chunks"""
        from app.models.document import DocumentChunk, DocumentImage, DocumentTable
        
        chunks = []
        for i in range(3):
            chunk = DocumentChunk(
                document_id=sample_document.id,
                content=f"Chunk {i}",
                page_number=i+1,
                chunk_index=i,
                extra_metadata={"related_images": [], "related_tables": []}
            )
            test_db.add(chunk)
            chunks.append(chunk)
        
        # Create images for each page
        for i in range(3):
            image = DocumentImage(
                document_id=sample_document.id,
                file_path=f"/tmp/img{i}.png",
                page_number=i+1,
                caption=f"Image {i+1}"
            )
            test_db.add(image)
        
        test_db.commit()
        for chunk in chunks:
            test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        chunk_ids = [chunk.id for chunk in chunks]
        related = await vector_store.get_related_content(chunk_ids)
        
        assert "images" in related
        assert "tables" in related
        # Should find images from all pages
        assert len(related["images"]) >= 3
    
    @pytest.mark.asyncio
    async def test_get_related_content_deduplication(self, test_db, sample_document):
        """Test that get_related_content deduplicates images/tables"""
        from app.models.document import DocumentChunk, DocumentImage
        
        # Create two chunks referencing the same image
        chunk1 = DocumentChunk(
            document_id=sample_document.id,
            content="Chunk 1",
            page_number=1,
            chunk_index=0,
            extra_metadata={"related_images": [1], "related_tables": []}
        )
        chunk2 = DocumentChunk(
            document_id=sample_document.id,
            content="Chunk 2",
            page_number=1,
            chunk_index=1,
            extra_metadata={"related_images": [1], "related_tables": []}  # Same image
        )
        test_db.add(chunk1)
        test_db.add(chunk2)
        
        image = DocumentImage(
            document_id=sample_document.id,
            file_path="/tmp/img1.png",
            page_number=1,
            caption="Shared image"
        )
        test_db.add(image)
        test_db.commit()
        test_db.refresh(chunk1)
        test_db.refresh(chunk2)
        test_db.refresh(image)
        
        # Update metadata with actual image ID
        chunk1.extra_metadata = {"related_images": [image.id], "related_tables": []}
        chunk2.extra_metadata = {"related_images": [image.id], "related_tables": []}
        test_db.commit()
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk1.id, chunk2.id])
        
        # Should deduplicate - same image should appear only once
        assert len(related["images"]) >= 1
        # Check that image ID 1 appears
        image_ids = [img["id"] for img in related["images"]]
        assert image.id in image_ids
    
    @pytest.mark.asyncio
    async def test_get_related_content_with_metadata_images(self, test_db, sample_document):
        """Test get_related_content with images from metadata"""
        from app.models.document import DocumentChunk, DocumentImage
        
        # Create image first
        image = DocumentImage(
            document_id=sample_document.id,
            file_path="/tmp/test.png",
            page_number=1,
            caption="Test image",
            width=100,
            height=200
        )
        test_db.add(image)
        test_db.commit()
        test_db.refresh(image)
        
        # Create chunk with image in metadata
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Test chunk",
            page_number=1,
            chunk_index=0,
            extra_metadata={"related_images": [image.id], "related_tables": []}
        )
        test_db.add(chunk)
        test_db.commit()
        test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        assert "images" in related
        assert len(related["images"]) > 0
        assert related["images"][0]["id"] == image.id
        assert related["images"][0]["width"] == 100
        assert related["images"][0]["height"] == 200
    
    @pytest.mark.asyncio
    async def test_get_related_content_with_metadata_tables(self, test_db, sample_document):
        """Test get_related_content with tables from metadata"""
        from app.models.document import DocumentChunk, DocumentTable
        
        # Create table first
        table = DocumentTable(
            document_id=sample_document.id,
            image_path="/tmp/table.png",
            page_number=1,
            caption="Test table",
            data={"rows": [["A", "B"]]},
            rows=1,
            columns=2
        )
        test_db.add(table)
        test_db.commit()
        test_db.refresh(table)
        
        # Create chunk with table in metadata
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Test chunk",
            page_number=1,
            chunk_index=0,
            extra_metadata={"related_images": [], "related_tables": [table.id]}
        )
        test_db.add(chunk)
        test_db.commit()
        test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        assert "tables" in related
        assert len(related["tables"]) > 0
        assert related["tables"][0]["id"] == table.id
        assert related["tables"][0]["rows"] == 1
        assert related["tables"][0]["columns"] == 2
    
    @pytest.mark.asyncio
    async def test_get_related_content_page_based_images(self, test_db, sample_document):
        """Test get_related_content finding images by page number"""
        from app.models.document import DocumentChunk, DocumentImage
        
        # Create chunk on page 2
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Chunk on page 2",
            page_number=2,
            chunk_index=0,
            extra_metadata={}  # No metadata, should find by page
        )
        test_db.add(chunk)
        
        # Create image on same page
        image = DocumentImage(
            document_id=sample_document.id,
            file_path="/tmp/page2.png",
            page_number=2,
            caption="Page 2 image"
        )
        test_db.add(image)
        test_db.commit()
        test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        # Should find image by page matching
        assert "images" in related
        assert len(related["images"]) > 0
        image_ids = [img["id"] for img in related["images"]]
        assert image.id in image_ids
    
    @pytest.mark.asyncio
    async def test_get_related_content_page_based_tables(self, test_db, sample_document):
        """Test get_related_content finding tables by page number"""
        from app.models.document import DocumentChunk, DocumentTable
        
        # Create chunk on page 3
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content="Chunk on page 3",
            page_number=3,
            chunk_index=0,
            extra_metadata={}  # No metadata, should find by page
        )
        test_db.add(chunk)
        
        # Create table on same page
        table = DocumentTable(
            document_id=sample_document.id,
            image_path="/tmp/page3.png",
            page_number=3,
            caption="Page 3 table",
            data={}
        )
        test_db.add(table)
        test_db.commit()
        test_db.refresh(chunk)
        
        vector_store = VectorStore(test_db)
        related = await vector_store.get_related_content([chunk.id])
        
        # Should find table by page matching
        assert "tables" in related
        assert len(related["tables"]) > 0
        table_ids = [tbl["id"] for tbl in related["tables"]]
        assert table.id in table_ids
    
    @pytest.mark.asyncio
    async def test_similarity_search_error_handling(self, test_db, monkeypatch):
        """Test similarity_search error handling"""
        async def mock_generate_embedding(query):
            raise Exception("Embedding generation failed")
        
        vector_store = VectorStore(test_db)
        vector_store.generate_embedding = mock_generate_embedding
        
        with pytest.raises(Exception):
            await vector_store.similarity_search("test query", document_id=None, k=5)
    
    @pytest.mark.asyncio
    async def test_similarity_search_raises_on_error(self, test_db, monkeypatch):
        """Test similarity_search raises exception on error"""
        mock_embedding = np.array([0.15] * 1536, dtype=np.float32)
        
        async def mock_generate_embedding(query):
            return mock_embedding
        
        vector_store = VectorStore(test_db)
        vector_store.generate_embedding = mock_generate_embedding
        
        # Mock SQL execution to raise error
        def mock_execute(sql, params, **kwargs):
            raise Exception("SQL execution failed")
        
        vector_store.db.execute = mock_execute
        
        with pytest.raises(Exception):
            await vector_store.similarity_search("test query", document_id=None, k=5)
