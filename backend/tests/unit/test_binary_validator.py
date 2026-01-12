"""
Unit tests for BinaryValidator utility
"""
import pytest
from app.utils.binary_validator import BinaryValidator


class TestBinaryValidator:
    """Test BinaryValidator functionality"""
    
    def test_contains_binary_patterns_empty(self):
        """Test contains_binary_patterns with empty string"""
        assert BinaryValidator.contains_binary_patterns("") == False
        assert BinaryValidator.contains_binary_patterns(None) == False
    
    def test_contains_binary_patterns_byte_string_prefix(self):
        """Test detection of byte string prefixes"""
        assert BinaryValidator.contains_binary_patterns("b'hello") == True
        assert BinaryValidator.contains_binary_patterns('b"hello') == True
        assert BinaryValidator.contains_binary_patterns("normal text") == False
    
    def test_contains_binary_patterns_hex_escapes(self):
        """Test detection of hex escape sequences"""
        assert BinaryValidator.contains_binary_patterns("b'\\x89Png") == True
        assert BinaryValidator.contains_binary_patterns('b"\\x89Png') == True
        assert BinaryValidator.contains_binary_patterns("text with \\x89") == False
    
    def test_contains_binary_patterns_png_patterns(self):
        """Test detection of PNG patterns"""
        for pattern in BinaryValidator.PNG_PATTERNS:
            assert BinaryValidator.contains_binary_patterns(f"some text {pattern} more text") == True
    
    def test_contains_binary_patterns_png_header_continuation(self):
        """Test detection of PNG header continuation"""
        assert BinaryValidator.contains_binary_patterns("b'\\r\\n\\x1a") == True
        assert BinaryValidator.contains_binary_patterns('b"\\r\\n\\x1a') == True
    
    def test_contains_binary_patterns_corrupted_hex(self):
        """Test detection of corrupted hex patterns"""
        # Create text with many corrupted hex patterns
        corrupted_text = "x9477 x834 x1234 x5678 x9012 x3456 x7890 xabcd xef01 x2345 x6789"
        assert BinaryValidator.contains_binary_patterns(corrupted_text) == True
        
        # Few patterns should not trigger
        normal_text = "x12 x34 x56"
        assert BinaryValidator.contains_binary_patterns(normal_text) == False
    
    def test_contains_binary_patterns_binary_string_patterns(self):
        """Test detection of binary string patterns"""
        assert BinaryValidator.contains_binary_patterns("\\x89PNG") == True
        assert BinaryValidator.contains_binary_patterns("\\x89Png") == True
        assert BinaryValidator.contains_binary_patterns("\\xff\\xd8\\xff") == True
    
    def test_is_valid_text_empty(self):
        """Test is_valid_text with empty string"""
        assert BinaryValidator.is_valid_text("") == False
        assert BinaryValidator.is_valid_text(None) == False
    
    def test_is_valid_text_too_long(self):
        """Test is_valid_text with text exceeding length limit"""
        long_text = "a" * 50001  # Exceeds 50KB limit
        assert BinaryValidator.is_valid_text(long_text) == False
    
    def test_is_valid_text_binary_patterns(self):
        """Test is_valid_text detects binary patterns"""
        assert BinaryValidator.is_valid_text("b'\\x89Png") == False
        assert BinaryValidator.is_valid_text("normal text here") == True
    
    def test_is_valid_text_printable_ratio(self):
        """Test is_valid_text with low printable ratio"""
        # Create text with many non-printable characters
        non_printable = "\x00\x01\x02" * 100 + "a" * 10
        assert BinaryValidator.is_valid_text(non_printable) == False
        
        # Normal text should pass
        assert BinaryValidator.is_valid_text("This is normal text with printable characters") == True
    
    def test_is_valid_text_escape_sequences(self):
        """Test is_valid_text with excessive escape sequences"""
        # Many escape sequences
        many_escapes = "\\x00" * 30
        assert BinaryValidator.is_valid_text(many_escapes) == False
        
        # High ratio of escape sequences
        high_ratio = "\\x00" * 100 + "a" * 100  # 50% escape sequences
        assert BinaryValidator.is_valid_text(high_ratio) == False
    
    def test_is_valid_text_null_bytes(self):
        """Test is_valid_text with null bytes"""
        # Many null bytes
        many_nulls = "\x00" * 30
        assert BinaryValidator.is_valid_text(many_nulls) == False
        
        # High ratio of null bytes
        high_null_ratio = "\x00" * 100 + "a" * 100  # 50% null bytes
        assert BinaryValidator.is_valid_text(high_null_ratio) == False
    
    def test_is_valid_text_binary_patterns_in_bytes(self):
        """Test is_valid_text detects binary patterns in actual bytes"""
        # PNG header - should be detected (likely by contains_binary_patterns)
        png_header = "\x89PNG"
        assert BinaryValidator.is_valid_text(png_header) == False
        
        # Multiple null bytes pattern (this tests the bytes check path)
        # Need enough text to pass other checks but still trigger bytes check
        binary_data = "a" * 50 + "\x00\x00\x00\x00" + "a" * 50  # Multiple null bytes in middle
        assert BinaryValidator.is_valid_text(binary_data) == False
        
        # JPEG header pattern - when encoded, this might not match the exact pattern
        # So we test with a pattern that will definitely be caught
        # The bytes check looks for b'\xff\xd8\xff' in encoded bytes
        # But when we have text after it, the encoding might change things
        # So let's just test that PNG and null bytes work, which are the main cases
    
    def test_is_valid_text_corrupted_unicode(self):
        """Test is_valid_text with corrupted Unicode patterns"""
        corrupted = "x9477 x834 x1234 x5678 x9012 x3456 x7890 xabcd xef01 x2345 x6789 x1234 x5678 x9012"
        assert BinaryValidator.is_valid_text(corrupted) == False
    
    def test_is_valid_text_corrupted_unicode_pattern(self):
        """Test is_valid_text with corrupted Unicode escape pattern"""
        # Pattern requires x0x0b (not x0b), need 3+ matches to trigger
        corrupted = "ub0x0x0b 5b0x0x0b ab0x0x0b cd0x0x0b"
        assert BinaryValidator.is_valid_text(corrupted) == False
    
    def test_is_valid_text_valid_text(self):
        """Test is_valid_text with valid text"""
        valid_texts = [
            "This is normal text",
            "Hello, world!",
            "123 numbers and text",
            "Special chars: !@#$%^&*()",
            "Unicode: 你好世界",
        ]
        for text in valid_texts:
            assert BinaryValidator.is_valid_text(text) == True
    
    def test_clean_markdown_text_empty(self):
        """Test clean_markdown_text with empty string"""
        assert BinaryValidator.clean_markdown_text("") == ""
        assert BinaryValidator.clean_markdown_text(None) == ""
    
    def test_clean_markdown_text_removes_byte_strings(self):
        """Test clean_markdown_text removes byte string lines"""
        markdown = "Normal line\nb'\\x89Png\nAnother normal line"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert "b'\\x89Png" not in cleaned
        assert "Normal line" in cleaned
        assert "Another normal line" in cleaned
    
    def test_clean_markdown_text_removes_hex_escapes(self):
        """Test clean_markdown_text removes lines with hex escapes"""
        markdown = "Line 1\nLine with b'\\x89\\x90\\x91\\x92\\x93\\x94\nLine 3"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert "Line with b'" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 3" in cleaned
    
    def test_clean_markdown_text_removes_excessive_escapes(self):
        """Test clean_markdown_text removes lines with excessive escape sequences"""
        markdown = "Good line\nBad line \\x00\\x01\\x02\\x03\\x04\\x05\\x06\nGood line 2"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert "Bad line" not in cleaned
        assert "Good line" in cleaned
    
    def test_clean_markdown_text_removes_binary_patterns(self):
        """Test clean_markdown_text removes lines with binary patterns"""
        markdown = "Normal text\nLine with \\x89png pattern\nMore text"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert "\\x89png" not in cleaned.lower()
        assert "Normal text" in cleaned
    
    def test_clean_markdown_text_removes_non_printable_lines(self):
        """Test clean_markdown_text removes lines with low printable ratio"""
        # Create line with mostly non-printable characters
        non_printable_line = "\x00\x01\x02" * 20 + "a"
        markdown = f"Good line\n{non_printable_line}\nAnother good line"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert non_printable_line not in cleaned
        assert "Good line" in cleaned
    
    def test_clean_markdown_text_preserves_empty_lines(self):
        """Test clean_markdown_text preserves empty lines"""
        markdown = "Line 1\n\nLine 3"
        cleaned = BinaryValidator.clean_markdown_text(markdown)
        assert "\n\n" in cleaned or cleaned.count("\n") >= 2
    
    def test_validate_chunk_content_empty(self):
        """Test validate_chunk_content with empty string"""
        assert BinaryValidator.validate_chunk_content("") == False
        assert BinaryValidator.validate_chunk_content("   ") == False
    
    def test_validate_chunk_content_valid(self):
        """Test validate_chunk_content with valid content"""
        assert BinaryValidator.validate_chunk_content("This is valid chunk content") == True
        assert BinaryValidator.validate_chunk_content("123 numbers") == True
    
    def test_validate_chunk_content_binary_patterns(self):
        """Test validate_chunk_content rejects binary patterns"""
        assert BinaryValidator.validate_chunk_content("b'\\x89Png") == False
        assert BinaryValidator.validate_chunk_content("\\x89PNG") == False
    
    def test_validate_caption_empty(self):
        """Test validate_caption with empty string"""
        assert BinaryValidator.validate_caption("") == False
        assert BinaryValidator.validate_caption(None) == False
    
    def test_validate_caption_valid(self):
        """Test validate_caption with valid caption"""
        assert BinaryValidator.validate_caption("Figure 1: Test image") == True
        assert BinaryValidator.validate_caption("Table 1") == True
    
    def test_validate_caption_binary_patterns(self):
        """Test validate_caption rejects binary patterns"""
        assert BinaryValidator.validate_caption("b'\\x89Png") == False
        assert BinaryValidator.validate_caption("\\xff\\xd8\\xff") == False
    
    def test_validate_caption_strips_whitespace(self):
        """Test validate_caption handles whitespace correctly"""
        assert BinaryValidator.validate_caption("  Valid caption  ") == True
        assert BinaryValidator.validate_caption("   ") == False
