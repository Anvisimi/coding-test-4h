"""
Binary data validation utility.

Centralizes all binary data detection and validation logic to avoid code duplication.
"""
import re
from typing import List


class BinaryValidator:
    """Utility class for detecting and filtering binary data from text."""
    
    # Binary patterns to detect
    BYTE_STRING_PATTERNS = [
        r"b['\"].*?\\x[0-9a-fA-F]{2}",  # Python byte strings with hex escapes
        r"b['\"]\\x89Png",  # PNG header in byte strings
        r"b['\"]\\r\\n\\x1a",  # PNG header continuation
    ]
    
    PNG_PATTERNS = [
        "b'\\x89Png", 'b"\\x89Png', "b'\\x89PNG", 'b"\\x89PNG',
        "b'\\x89png", 'b"\\x89png',
        "b'\\x89Png\\r", 'b"\\x89Png\\r',
        "b'\\x89Png\\r\\n", 'b"\\x89Png\\r\\n',
        "b'\\x89Png\\r\\n\\x1a", 'b"\\x89Png\\r\\n\\x1a',
        "b'\\x89Png\\r\\n\\x1a\\n", 'b"\\x89Png\\r\\n\\x1a\\n',
    ]
    
    BINARY_STRING_PATTERNS = [
        "\\x89PNG", "\\x89Png", "\\xff\\xd8\\xff",  # JPEG header
        "\\x00\\x00\\x00", "\\r\\n\\x1a\\n",
    ]
    
    CORRUPTED_HEX_PATTERNS = [
        r'x[0-9a-fA-F]{3,}',  # x9477, x834, etc.
        r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b',  # ub0x0b, 5b0x0b
    ]
    
    @staticmethod
    def contains_binary_patterns(text: str) -> bool:
        """
        Check if text contains binary data patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if binary patterns detected, False otherwise
        """
        if not text:
            return False
        
        text_str = str(text)
        
        # Check for Python byte string representations
        if text_str.startswith("b'") or text_str.startswith('b"'):
            return True
        
        # Check for byte string patterns anywhere in text
        if "b'\\x" in text_str or 'b"\\x' in text_str:
            return True
        
        # Check PNG-specific patterns
        for pattern in BinaryValidator.PNG_PATTERNS:
            if pattern in text_str:
                return True
        
        # Check for PNG header continuation
        if "b'\\r\\n\\x1a" in text_str or 'b"\\r\\n\\x1a' in text_str:
            return True
        
        # Check for corrupted hex patterns
        for pattern in BinaryValidator.CORRUPTED_HEX_PATTERNS:
            if re.search(pattern, text_str, re.IGNORECASE):
                matches = re.findall(pattern, text_str, re.IGNORECASE)
                if len(matches) > 5:
                    return True
        
        # Check for binary string patterns
        text_lower = text_str.lower()
        binary_count = 0
        for pattern in BinaryValidator.BINARY_STRING_PATTERNS:
            count = text_lower.count(pattern.lower())
            if "png" in pattern.lower() or "jpeg" in pattern.lower() or "\\xff\\xd8" in pattern.lower():
                if count >= 1:
                    return True
            binary_count += count
        
        if binary_count > 3:
            return True
        
        return False
    
    @staticmethod
    def is_valid_text(text: str, min_printable_ratio: float = 0.8) -> bool:
        """
        Comprehensive text validation to filter out binary data.
        
        Args:
            text: Text to validate
            min_printable_ratio: Minimum ratio of printable characters (default 0.8)
            
        Returns:
            True if text appears to be valid readable text
        """
        if not text or len(text) == 0:
            return False
        
        # Check length limit
        if len(text) > 50000:  # 50KB limit
            return False
        
        # Quick check for binary patterns first (most common case)
        if BinaryValidator.contains_binary_patterns(text):
            return False
        
        # Count printable characters
        printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
        printable_ratio = printable_count / len(text) if len(text) > 0 else 0
        
        if printable_ratio < min_printable_ratio:
            return False
        
        # Check for excessive escape sequences
        escape_seq_count = text.count('\\x')
        if escape_seq_count > 0:
            escape_ratio = escape_seq_count / len(text) if len(text) > 0 else 0
            if escape_ratio > 0.02:  # More than 2% escape sequences
                return False
            if escape_seq_count > 20:
                return False
        
        # Check for excessive null bytes
        null_byte_count = text.count('\x00')
        if null_byte_count > 0:
            null_ratio = null_byte_count / len(text) if len(text) > 0 else 0
            if null_ratio > 0.02 or null_byte_count > 20:
                return False
        
        # Check for binary patterns in actual bytes
        try:
            text_bytes = text.encode('utf-8', errors='ignore')[:200]
            binary_patterns = [
                b'\x89PNG',  # PNG header
                b'\xff\xd8\xff',  # JPEG header
                b'\x00\x00\x00\x00',  # Multiple null bytes
            ]
            for pattern in binary_patterns:
                if pattern in text_bytes:
                    return False
        except:
            pass
        
        # Check for corrupted Unicode escape sequences
        corrupted_hex_pattern = re.compile(r'x[0-9a-fA-F]{2,}')
        corrupted_hex_matches = corrupted_hex_pattern.findall(text)
        if len(corrupted_hex_matches) > 5:
            total_corrupted_length = sum(len(m) for m in corrupted_hex_matches)
            if len(text) > 100 and total_corrupted_length / len(text) > 0.05:
                return False
            if len(corrupted_hex_matches) > 20:
                return False
        
        # Check for corrupted Unicode patterns
        corrupted_unicode_pattern = re.compile(r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b', re.IGNORECASE)
        unicode_corrupted_matches = corrupted_unicode_pattern.findall(text)
        if len(unicode_corrupted_matches) > 2:
            return False
        
        return True
    
    @staticmethod
    def clean_markdown_text(text: str) -> str:
        """
        Clean markdown text by removing binary data sections.
        
        Args:
            text: Markdown text to clean
            
        Returns:
            Cleaned text with binary data removed
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Skip lines that look like binary data
            if line_stripped.startswith("b'") or line_stripped.startswith('b"'):
                removed_count += 1
                continue
            
            if "b'\\x" in line_stripped or 'b"\\x' in line_stripped:
                removed_count += 1
                continue
            
            # Check for excessive escape sequences
            escape_count = line_stripped.count('\\x')
            if escape_count > 5:
                removed_count += 1
                continue
            
            # Check for binary patterns
            line_lower = line_stripped.lower()
            if '\\x89png' in line_lower or '\\xff\\xd8\\xff' in line_lower:
                removed_count += 1
                continue
            
            # Check if line is mostly non-printable
            printable_count = sum(1 for c in line_stripped if c.isprintable() or c.isspace())
            if len(line_stripped) > 0 and printable_count / len(line_stripped) < 0.6:
                removed_count += 1
                continue
            
            # Line looks okay, keep it
            cleaned_lines.append(line)
        
        if removed_count > 0:
            print(f"Cleaned markdown: removed {removed_count} lines containing binary data")
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def validate_chunk_content(content: str) -> bool:
        """
        Validate chunk content before saving.
        
        Args:
            content: Chunk content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        if not content or not content.strip():
            return False
        
        # Check for binary patterns
        if BinaryValidator.contains_binary_patterns(content):
            return False
        
        # Final validation check
        return BinaryValidator.is_valid_text(content)
    
    @staticmethod
    def validate_caption(caption: str) -> bool:
        """
        Validate caption text before saving.
        
        Args:
            caption: Caption text to validate
            
        Returns:
            True if caption is valid, False otherwise
        """
        if not caption:
            return False
        
        caption_str = str(caption).strip()
        
        # Check for binary patterns
        if BinaryValidator.contains_binary_patterns(caption_str):
            return False
        
        # Final validation
        return BinaryValidator.is_valid_text(caption_str)

