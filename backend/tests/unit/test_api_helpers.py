"""
Unit tests for API helper functions and error handling
"""
import pytest
from fastapi import HTTPException
from app.models.conversation import Conversation, Message
from app.models.document import Document
from datetime import datetime, timezone


class TestAPIHelpers:
    """Test API helper functions"""
    
    def test_conversation_creation(self, test_db):
        """Test creating a conversation"""
        conv = Conversation(
            title="Test Conversation",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        test_db.add(conv)
        test_db.commit()
        test_db.refresh(conv)
        
        assert conv.id is not None
        assert conv.title == "Test Conversation"
    
    def test_message_creation(self, test_db, sample_conversation):
        """Test creating a message"""
        msg = Message(
            conversation_id=sample_conversation.id,
            role="user",
            content="Test message",
            sources=[]
        )
        test_db.add(msg)
        test_db.commit()
        test_db.refresh(msg)
        
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "Test message"
    
    def test_document_creation(self, test_db):
        """Test creating a document"""
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            processing_status="pending"
        )
        test_db.add(doc)
        test_db.commit()
        test_db.refresh(doc)
        
        assert doc.id is not None
        assert doc.filename == "test.pdf"
        assert doc.processing_status == "pending"
    
    def test_conversation_with_messages(self, test_db):
        """Test conversation with multiple messages"""
        conv = Conversation(
            title="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        test_db.add(conv)
        test_db.commit()
        test_db.refresh(conv)
        
        # Add messages
        for i in range(3):
            msg = Message(
                conversation_id=conv.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i+1}"
            )
            test_db.add(msg)
        
        test_db.commit()
        test_db.refresh(conv)
        
        assert len(conv.messages) == 3
    
    def test_document_status_updates(self, test_db):
        """Test document status updates"""
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            processing_status="pending"
        )
        test_db.add(doc)
        test_db.commit()
        test_db.refresh(doc)
        
        assert doc.processing_status == "pending"
        
        # Update status
        doc.processing_status = "completed"
        doc.total_pages = 10
        doc.text_chunks_count = 50
        test_db.commit()
        test_db.refresh(doc)
        
        assert doc.processing_status == "completed"
        assert doc.total_pages == 10
        assert doc.text_chunks_count == 50

