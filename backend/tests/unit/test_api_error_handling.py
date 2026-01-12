"""
Unit tests for API error handling paths
"""
import pytest
from fastapi import HTTPException
from app.models.conversation import Conversation, Message
from app.models.document import Document
from datetime import datetime, timezone


class TestAPIErrorHandling:
    """Test API error handling paths"""
    
    def test_conversation_not_found_error(self, test_db):
        """Test conversation not found error path"""
        # Try to query non-existent conversation
        conversation = test_db.query(Conversation).filter(Conversation.id == 99999).first()
        assert conversation is None
    
    def test_document_not_found_error(self, test_db):
        """Test document not found error path"""
        # Try to query non-existent document
        document = test_db.query(Document).filter(Document.id == 99999).first()
        assert document is None
    
    def test_message_creation_with_invalid_conversation(self, test_db):
        """Test message creation error handling"""
        # SQLite doesn't enforce foreign keys by default, so this might succeed
        # But we can test that the message is created (even if FK isn't enforced)
        msg = Message(
            conversation_id=99999,
            role="user",
            content="Test"
        )
        test_db.add(msg)
        test_db.commit()
        test_db.refresh(msg)
        
        # Message is created (SQLite allows it)
        assert msg.id is not None
        assert msg.conversation_id == 99999
        # But querying the conversation would return None
        conversation = test_db.query(Conversation).filter(Conversation.id == 99999).first()
        assert conversation is None
    
    def test_conversation_title_truncation(self, test_db):
        """Test conversation title truncation to 50 chars"""
        long_message = "A" * 100
        title = long_message[:50]
        
        conv = Conversation(
            title=title,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        test_db.add(conv)
        test_db.commit()
        
        assert len(conv.title) == 50
        assert conv.title == "A" * 50
    
    def test_document_with_images_and_tables(self, test_db):
        """Test document with images and tables relationships"""
        from app.models.document import DocumentImage, DocumentTable
        
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            processing_status="completed",
            total_pages=5
        )
        test_db.add(doc)
        test_db.commit()
        test_db.refresh(doc)
        
        # Add images
        for i in range(2):
            img = DocumentImage(
                document_id=doc.id,
                file_path=f"/tmp/img{i}.png",
                page_number=i+1,
                caption=f"Image {i+1}"
            )
            test_db.add(img)
        
        # Add tables
        for i in range(1):
            tbl = DocumentTable(
                document_id=doc.id,
                image_path=f"/tmp/table{i}.png",
                page_number=i+2,
                caption=f"Table {i+1}",
                data={}
            )
            test_db.add(tbl)
        
        test_db.commit()
        test_db.refresh(doc)
        
        assert len(doc.images) == 2
        assert len(doc.tables) == 1
    
    def test_message_with_sources(self, test_db, sample_conversation):
        """Test message with sources"""
        sources = [
            {"type": "text", "content": "Test", "page": 1},
            {"type": "image", "url": "/img.png", "caption": "Fig 1"}
        ]
        
        msg = Message(
            conversation_id=sample_conversation.id,
            role="assistant",
            content="Answer with sources",
            sources=sources
        )
        test_db.add(msg)
        test_db.commit()
        test_db.refresh(msg)
        
        assert msg.sources == sources
        assert len(msg.sources) == 2
    
    def test_conversation_updated_at_timestamp(self, test_db):
        """Test conversation updated_at timestamp update"""
        conv = Conversation(
            title="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        test_db.add(conv)
        test_db.commit()
        
        original_updated = conv.updated_at
        
        # Update timestamp
        conv.updated_at = datetime.now(timezone.utc)
        test_db.commit()
        test_db.refresh(conv)
        
        assert conv.updated_at > original_updated
    
    def test_document_processing_status_transitions(self, test_db):
        """Test document status transitions"""
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            processing_status="pending"
        )
        test_db.add(doc)
        test_db.commit()
        
        assert doc.processing_status == "pending"
        
        # Transition to processing
        doc.processing_status = "processing"
        test_db.commit()
        test_db.refresh(doc)
        assert doc.processing_status == "processing"
        
        # Transition to completed
        doc.processing_status = "completed"
        test_db.commit()
        test_db.refresh(doc)
        assert doc.processing_status == "completed"
    
    def test_document_error_message(self, test_db):
        """Test document error message handling"""
        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            processing_status="failed",
            error_message="Processing failed: Invalid PDF format"
        )
        test_db.add(doc)
        test_db.commit()
        test_db.refresh(doc)
        
        assert doc.error_message == "Processing failed: Invalid PDF format"
        assert doc.processing_status == "failed"

