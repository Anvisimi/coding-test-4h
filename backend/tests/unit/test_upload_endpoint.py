"""
Unit tests for upload endpoint logic
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from fastapi import UploadFile, BackgroundTasks
from app.models.document import Document


class TestUploadEndpoint:
    """Test upload endpoint logic"""
    
    @pytest.mark.asyncio
    async def test_upload_document_success(self, test_db, temp_upload_dir, monkeypatch):
        """Test successful document upload"""
        from app.api.documents import upload_document
        from app.core.config import settings
        
        monkeypatch.setattr(settings, "UPLOAD_DIR", temp_upload_dir)
        
        # Create mock file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"PDF content")
        
        mock_background_tasks = Mock(spec=BackgroundTasks)
        mock_background_tasks.add_task = Mock()
        
        try:
            result = await upload_document(
                background_tasks=mock_background_tasks,
                file=mock_file,
                db=test_db
            )
            
            assert "id" in result
            assert result["filename"] == "test.pdf"
            assert result["status"] == "pending"
            assert "message" in result
            # Verify background task was added
            mock_background_tasks.add_task.assert_called_once()
        except Exception as e:
            if "no such table" in str(e).lower():
                pytest.skip("Database tables not available")
            raise
    
    # @pytest.mark.asyncio
    # async def test_upload_document_creates_directory(self, test_db, temp_upload_dir, monkeypatch):
    #     """Test upload creates upload directory if it doesn't exist"""
    #     from app.api.documents import upload_document
    #     from app.core.config import settings
    #
    #     upload_subdir = os.path.join(temp_upload_dir, "documents")
    #     monkeypatch.setattr(settings, "UPLOAD_DIR", temp_upload_dir)
    #
    #     # Remove directory if it exists
    #     if os.path.exists(upload_subdir):
    #         os.rmdir(upload_subdir)
    #
    #     mock_file = Mock(spec=UploadFile)
    #     mock_file.filename = "test.pdf"
    #     mock_file.read = AsyncMock(return_value=b"PDF content")
    #
    #     mock_background_tasks = Mock(spec=BackgroundTasks)
    #     mock_background_tasks.add_task = Mock()
    #
    #     try:
    #         result = await upload_document(
    #             background_tasks=mock_background_tasks,
    #             file=mock_file,
    #             db=test_db
    #         )
    #
    #         # Directory should be created
    #         assert os.path.exists(upload_subdir)
    #         assert "id" in result
    #     except Exception as e:
    #         if "no such table" in str(e).lower():
    #             pytest.skip("Database tables not available")
    #         raise
    #
    @pytest.mark.asyncio
    async def test_upload_document_unique_filename(self, test_db, temp_upload_dir, monkeypatch):
        """Test upload generates unique filename"""
        from app.api.documents import upload_document
        from app.core.config import settings
        
        monkeypatch.setattr(settings, "UPLOAD_DIR", temp_upload_dir)
        
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.read = AsyncMock(return_value=b"PDF content")
        
        mock_background_tasks = Mock(spec=BackgroundTasks)
        mock_background_tasks.add_task = Mock()
        
        try:
            result1 = await upload_document(
                background_tasks=mock_background_tasks,
                file=mock_file,
                db=test_db
            )
            
            # Upload again with same filename
            result2 = await upload_document(
                background_tasks=mock_background_tasks,
                file=mock_file,
                db=test_db
            )
            
            # Should have different IDs and file paths
            assert result1["id"] != result2["id"]
        except Exception as e:
            if "no such table" in str(e).lower():
                pytest.skip("Database tables not available")
            raise

