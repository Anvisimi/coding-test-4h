"""
Pytest configuration and shared fixtures
"""
# IMPORTANT: Patch Vector BEFORE any other imports
import tests._patch_vector  # noqa: F401

import pytest
import os
import tempfile
import shutil
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from app.db.session import Base, get_db
from app.core.config import settings
from datetime import datetime, timezone

# Import models after patching
from app.models import document, conversation


# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def test_db():
    """Create a test database session"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    
    # Always use manual table creation for SQLite compatibility
    from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, ForeignKey, JSON, Text
    metadata = MetaData()
    
    Table('documents', metadata,
        Column('id', Integer, primary_key=True),
        Column('filename', String, nullable=False),
        Column('file_path', String, nullable=False),
        Column('upload_date', DateTime),
        Column('processing_status', String),
        Column('error_message', Text),
        Column('total_pages', Integer),
        Column('text_chunks_count', Integer),
        Column('images_count', Integer),
        Column('tables_count', Integer),
    )
    
    Table('document_chunks', metadata,
        Column('id', Integer, primary_key=True),
        Column('document_id', Integer, ForeignKey('documents.id', ondelete='CASCADE')),
        Column('content', Text, nullable=False),
        Column('embedding', Text),  # Text instead of Vector for SQLite
        Column('page_number', Integer),
        Column('chunk_index', Integer),
        Column('extra_metadata', JSON),
        Column('created_at', DateTime),
    )
    
    Table('document_images', metadata,
        Column('id', Integer, primary_key=True),
        Column('document_id', Integer, ForeignKey('documents.id', ondelete='CASCADE')),
        Column('file_path', String, nullable=False),
        Column('page_number', Integer),
        Column('caption', Text),
        Column('width', Integer),
        Column('height', Integer),
        Column('extra_metadata', JSON),
        Column('created_at', DateTime),
    )
    
    Table('document_tables', metadata,
        Column('id', Integer, primary_key=True),
        Column('document_id', Integer, ForeignKey('documents.id', ondelete='CASCADE')),
        Column('image_path', String, nullable=False),
        Column('data', JSON),
        Column('page_number', Integer),
        Column('caption', Text),
        Column('rows', Integer),
        Column('columns', Integer),
        Column('extra_metadata', JSON),
        Column('created_at', DateTime),
    )
    
    Table('conversations', metadata,
        Column('id', Integer, primary_key=True),
        Column('title', String),
        Column('created_at', DateTime),
        Column('updated_at', DateTime),
        Column('document_id', Integer, ForeignKey('documents.id', ondelete='SET NULL')),
    )
    
    Table('messages', metadata,
        Column('id', Integer, primary_key=True),
        Column('conversation_id', Integer, ForeignKey('conversations.id', ondelete='CASCADE')),
        Column('role', String, nullable=False),
        Column('content', Text, nullable=False),
        Column('sources', JSON),
        Column('created_at', DateTime),
    )
    
    metadata.create_all(bind=engine)
    
    # Verify tables were created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    created_tables = inspector.get_table_names()
    assert 'documents' in created_tables, f"documents table not created. Tables: {created_tables}"
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_client(test_db):
    """Create a test client with database override"""
    # Import app here AFTER Vector is patched
    from app.main import app
    
    # Ensure test_db has the bind attribute set correctly
    if not hasattr(test_db, 'bind'):
        # Get the engine from the session
        test_db.bind = test_db.get_bind()
    
    def override_get_db():
        # Return the same test_db session that has tables
        yield test_db
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def temp_upload_dir():
    """Create a temporary directory for uploads"""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(f"{temp_dir}/documents", exist_ok=True)
    os.makedirs(f"{temp_dir}/images", exist_ok=True)
    os.makedirs(f"{temp_dir}/tables", exist_ok=True)
    
    original_upload_dir = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = temp_dir
    
    yield temp_dir
    
    settings.UPLOAD_DIR = original_upload_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_document(test_db):
    """Create a sample document"""
    from app.models.document import Document
    
    doc = Document(
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        upload_date=datetime.now(timezone.utc),
        processing_status="completed",
        total_pages=5,
        text_chunks_count=10,
        images_count=2,
        tables_count=1
    )
    test_db.add(doc)
    test_db.commit()
    test_db.refresh(doc)
    return doc


@pytest.fixture
def sample_document_chunks(test_db, sample_document):
    """Create sample document chunks"""
    from app.models.document import DocumentChunk
    import numpy as np
    
    chunks = []
    for i in range(3):
        embedding = np.random.rand(1536).astype(np.float32).tolist()
        chunk = DocumentChunk(
            document_id=sample_document.id,
            content=f"Chunk {i+1}",
            embedding=embedding,
            page_number=i+1,
            chunk_index=i
        )
        test_db.add(chunk)
        chunks.append(chunk)
    
    test_db.commit()
    for chunk in chunks:
        test_db.refresh(chunk)
    return chunks


@pytest.fixture
def sample_conversation(test_db, sample_document):
    """Create a sample conversation"""
    from app.models.conversation import Conversation
    
    conv = Conversation(
        title="Test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        document_id=sample_document.id
    )
    test_db.add(conv)
    test_db.commit()
    test_db.refresh(conv)
    return conv


@pytest.fixture
def mock_openai_chat(monkeypatch):
    """Mock OpenAI chat API"""
    class MockClient:
        class MockChat:
            class MockCompletions:
                @staticmethod
                def create(*args, **kwargs):
                    class MockResponse:
                        class MockChoice:
                            class MockMessage:
                                content = "Test response"
                            message = MockMessage()
                        choices = [MockChoice()]
                    return MockResponse()
            completions = MockCompletions()
        chat = MockChat()
    
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: MockClient())
