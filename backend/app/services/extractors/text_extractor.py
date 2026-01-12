"""
Text extraction and chunking service.

DONE: Implemented text extraction from Docling and PyMuPDF
DONE: Implemented intelligent text chunking with overlap
DONE: Implemented binary data validation and filtering
DONE: Implemented chunk storage with embeddings
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.services.vector_store import VectorStore
from app.core.config import settings
from app.utils.binary_validator import BinaryValidator
import fitz  # PyMuPDF
import re
import asyncio


class TextExtractor:
    """
    Extract and chunk text from PDF documents.
    
    Handles text extraction from both Docling and PyMuPDF sources,
    validates text to filter binary data, and chunks text intelligently
    for vector storage.
    """
    
    def __init__(self, db: Session, vector_store: VectorStore, binary_validator: BinaryValidator):
        self.db = db
        self.vector_store = vector_store
        self.binary_validator = binary_validator
    
    async def extract_and_save_text(
        self,
        doc: Any,
        file_path: str,
        pages: List[Any],
        document_id: int
    ) -> int:
        """
        Extract text from document and save as chunks.
        
        Strategy:
        1. Try full document text extraction (with validation)
        2. Fallback to page-by-page extraction (more reliable)
        3. Use PyMuPDF as primary source, Docling as fallback
        4. Validate and filter binary data at each step
        
        Args:
            doc: Docling document object
            file_path: Path to PDF file
            pages: List of page objects from Docling
            document_id: Document ID for database storage
            
        Returns:
            Number of text chunks created
        """
        text_chunks_count = 0
        
        # Step 1: Try full document text extraction (with strict validation)
        # This is faster but less reliable due to potential binary data contamination
        full_text = self._extract_full_text(doc)
        if full_text:
            chunks = self._chunk_text(full_text, document_id, 1)
            if chunks:
                await self._save_text_chunks(chunks, document_id)
                text_chunks_count += len(chunks)
                print(f"Created {len(chunks)} chunks from full text")
        
        # Step 2: Page-by-page extraction (more reliable, cleaner)
        # Open PDF with PyMuPDF for reliable text extraction (CPU-intensive, run in thread)
        pdf_doc_for_text = None
        try:
            loop = asyncio.get_event_loop()
            pdf_doc_for_text = await loop.run_in_executor(None, fitz.open, file_path)
            print(f"Opened PDF with PyMuPDF for text extraction: {len(pdf_doc_for_text)} pages")
        except Exception as e:
            print(f"Error opening PDF with PyMuPDF for text extraction: {e}")
        
        # Process each page individually for better control and validation
        for page_idx, page in enumerate(pages, start=1):
            page_num = self._get_page_number(page, page_idx)
            
            # Extract text from page using multiple strategies
            page_text = self._extract_page_text(page, page_num, pdf_doc_for_text)
            
            if not page_text:
                continue
            
            # Validate and chunk page text
            page_text_str = str(page_text).strip()
            if not page_text_str:
                continue
            
            # Binary validation: Skip pages with binary data contamination
            if self.binary_validator.contains_binary_patterns(page_text_str):
                print(f"Skipping page {page_num} text - contains binary patterns")
                continue
            
            if not self.binary_validator.is_valid_text(page_text_str):
                print(f"Skipping page {page_num} text extraction - appears to be binary data")
                continue
            
            # Chunk the validated text
            chunks = self._chunk_text(page_text_str, document_id, page_num)
            if chunks:
                # Final validation: filter out any chunks that still contain binary data
                clean_chunks = [
                    chunk for chunk in chunks
                    if not self.binary_validator.contains_binary_patterns(chunk.get("content", ""))
                ]
                
                if clean_chunks:
                    await self._save_text_chunks(clean_chunks, document_id)
                    text_chunks_count += len(clean_chunks)
                    if page_num <= 5:
                        print(f"Created {len(clean_chunks)} chunks from page {page_num} text")
        
        # Clean up PyMuPDF document
        if pdf_doc_for_text:
            try:
                pdf_doc_for_text.close()
            except:
                pass
        
        return text_chunks_count
    
    def _extract_full_text(self, doc: Any) -> Optional[str]:
        """
        Extract full document text from Docling document.
        
        Uses multiple validation layers to detect and filter binary data:
        1. Pattern matching for hex corruption
        2. Binary header detection
        3. Sample validation
        4. Length-based heuristics
        
        Returns None if text appears contaminated with binary data.
        """
        full_text = ""
        
        # Try export_to_markdown first (most complete)
        try:
            if hasattr(doc, 'export_to_markdown'):
                full_text = doc.export_to_markdown()
                print(f"Got full text via export_to_markdown: {len(full_text)} chars")
        except Exception as e:
            print(f"Error in export_to_markdown: {e}")
        
        # Fallback: try direct text attribute
        if not full_text:
            try:
                if hasattr(doc, 'text'):
                    full_text = doc.text
                    print(f"Got text directly: {len(full_text) if full_text else 0} chars")
            except Exception as e:
                print(f"Error getting text: {e}")
        
        if not full_text or not full_text.strip():
            return None
        
        full_text_str = str(full_text).strip()
        
        # Validation Layer 1: Pattern matching for corrupted hex patterns
        # These patterns indicate binary data contamination in text
        corrupted_patterns = [
            r'x[0-9a-fA-F]{3,}',  # x9477, x834, etc.
            r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b',  # ub0x0b, 5b0x0b pattern
        ]
        
        found_binary = False
        for pattern in corrupted_patterns:
            matches = re.findall(pattern, full_text_str, re.IGNORECASE)
            if len(matches) > 0:  # Even 1 match is suspicious
                print(f"Skipping full text extraction - found {len(matches)} instances of corrupted hex pattern")
                found_binary = True
                break
        
        # Validation Layer 2: Binary header detection
        if not found_binary:
            if self.binary_validator.contains_binary_patterns(full_text_str):
                print(f"Skipping full text extraction - contains binary patterns")
                found_binary = True
            elif any(header.lower() in full_text_str.lower() for header in ['\\x89png', '\\xff\\xd8\\xff', 'lhdr']):
                print(f"Skipping full text extraction - contains binary header patterns")
                found_binary = True
        
        # Validation Layer 3: Sample validation (check first 5KB)
        if not found_binary:
            sample_text = full_text_str[:5000]
            if not self.binary_validator.is_valid_text(sample_text):
                print(f"Skipping full text extraction - sample validation failed")
                found_binary = True
        
        if found_binary:
            print("Skipping full text extraction - will rely on page-by-page extraction only (cleaner)")
            return None
        
        # Validation Layer 4: Length-based heuristic
        # Very long text (>20KB) might contain hidden binary data
        if len(full_text_str) > 20000:
            print(f"Skipping full text extraction - text is very long ({len(full_text_str)} chars), may contain hidden binary data")
            return None
        
        # Final validation and cleaning
        full_text_str = self.binary_validator.clean_markdown_text(full_text_str)
        if self.binary_validator.is_valid_text(full_text_str):
            return full_text_str
        
        print("Skipping full text extraction - validation failed after cleaning")
        return None
    
    def _extract_page_text(self, page: Any, page_num: int, pdf_doc_for_text: Optional[Any]) -> Optional[str]:
        """
        Extract text from a single page using multiple strategies.
        
        Priority order:
        1. PyMuPDF (most reliable for actual PDF text)
        2. Docling page object attributes
        3. Docling page items aggregation
        
        Args:
            page: Docling page object
            page_num: Page number (1-indexed)
            pdf_doc_for_text: PyMuPDF document object (optional)
            
        Returns:
            Extracted text or None
        """
        page_text = None
        
        # Strategy 1: PyMuPDF extraction (most reliable)
        # PyMuPDF reads directly from PDF, avoiding Docling's potential binary contamination
        if pdf_doc_for_text and page_num <= len(pdf_doc_for_text):
            try:
                pdf_page = pdf_doc_for_text[page_num - 1]  # PyMuPDF is 0-indexed
                page_text = pdf_page.get_text()
                if page_text and page_text.strip():
                    if page_num <= 3:
                        print(f"DEBUG: Extracted {len(page_text)} chars from page {page_num} via PyMuPDF")
            except Exception as e:
                if page_num <= 3:
                    print(f"DEBUG: Error extracting text from PyMuPDF page {page_num}: {e}")
        
        # Strategy 2: Docling page object attributes (fallback)
        if not page_text:
            if hasattr(page, 'text'):
                page_text = page.text
            elif hasattr(page, 'get_text'):
                page_text = page.get_text()
            elif hasattr(page, 'content'):
                page_text = page.content
            elif hasattr(page, 'export_to_markdown'):
                try:
                    page_text = page.export_to_markdown()
                except:
                    pass
            elif isinstance(page, str):
                page_text = page
        
        # Strategy 3: Aggregate from page items (last resort)
        if not page_text and hasattr(page, 'items'):
            page_text_parts = []
            for item in page.items:
                if hasattr(item, 'text') and item.text:
                    item_text = str(item.text)
                    # Validate each item before adding
                    if self.binary_validator.is_valid_text(item_text):
                        page_text_parts.append(item_text)
            if page_text_parts:
                page_text = '\n\n'.join(page_text_parts)
        
        return page_text
    
    def _get_page_number(self, page: Any, default: int) -> int:
        """Extract page number from page object."""
        if hasattr(page, 'page'):
            return page.page
        elif hasattr(page, 'page_number'):
            return page.page_number
        return default
    
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        DONE: Implemented intelligent text chunking strategy
        - Split by paragraphs (natural boundaries)
        - Fallback to sentence-level splitting
        - Character-level splitting for very long sentences
        - Maintains context with overlap (200 chars default)
        - Respects minimum chunk size (600 chars) at natural boundaries
        
        Chunking Algorithm:
        1. Split by paragraphs (double newlines) - preserves semantic meaning
        2. For paragraphs that exceed chunk_size, split by sentences
        3. For sentences that exceed chunk_size, split by characters
        4. Create chunks at natural boundaries (paragraph/sentence end) when minimum size reached
        5. Add overlap from previous chunk to maintain context
        
        This strategy creates ~5-7 chunks per page, targeting 80-100 total chunks
        for a 15-page document, which is optimal for RAG retrieval.
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or not text.strip():
            return []
        
        # Validate text - skip if it's binary data
        if not self.binary_validator.is_valid_text(text):
            print(f"Skipping invalid text on page {page_number} (likely binary data, length: {len(text)})")
            return []
        
        # Limit text length to prevent extremely long extractions
        max_text_length = 50000  # 50KB limit per page
        if len(text) > max_text_length:
            print(f"Truncating text on page {page_number} from {len(text)} to {max_text_length} characters")
            text = text[:max_text_length]
        
        chunks = []
        chunk_size = settings.CHUNK_SIZE  # 1000 chars
        chunk_overlap = settings.CHUNK_OVERLAP  # 200 chars
        
        # Minimum chunk size at natural boundaries (60% of chunk_size)
        # This ensures chunks aren't too small while allowing more frequent chunking
        # Target: ~5-7 chunks per page to reach 80-100 total for 15 pages
        min_chunk_size = int(chunk_size * 0.6)  # 600 chars minimum at natural boundaries
        
        # Helper function to split text by separator
        def _split_text_by_separator(text: str, separator: str) -> List[str]:
            """Split text by separator and clean up."""
            splits = text.split(separator)
            return [s.strip() for s in splits if s.strip()]
        
        # Helper function to get overlap text from chunk
        def _get_overlap_text(chunk_text: str, overlap_size: int) -> str:
            """
            Get the last N characters for overlap, preferring word boundaries.
            
            This maintains context between chunks by including the end of the
            previous chunk at the start of the next chunk.
            """
            if len(chunk_text) <= overlap_size:
                return chunk_text
            # Try to get overlap at word boundary for better semantic continuity
            overlap_start = len(chunk_text) - overlap_size
            space_idx = chunk_text.rfind(' ', 0, overlap_start)
            if space_idx > overlap_start - 100:  # If space is reasonably close
                return chunk_text[space_idx + 1:]
            return chunk_text[overlap_start:]
        
        # Helper function to create chunk if it meets minimum size
        def _create_chunk_if_ready(chunk_text: str) -> bool:
            """Create chunk if it meets minimum size and is valid. Returns True if chunk was created."""
            if not chunk_text or len(chunk_text) < min_chunk_size:
                return False
            
            if self.binary_validator.is_valid_text(chunk_text) and not self.binary_validator.contains_binary_patterns(chunk_text):
                chunks.append({
                    "content": chunk_text,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "metadata": {}
                })
                return True
            return False
        
        # Step 1: Split by paragraphs (double newlines) - preserves semantic meaning
        paragraphs = _split_text_by_separator(text, '\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed chunk_size
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                # Paragraph fits - add it to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                
                # Create chunk at natural boundary (end of paragraph) if minimum size reached
                # This creates more chunks while respecting semantic boundaries
                if len(current_chunk) >= min_chunk_size:
                    if _create_chunk_if_ready(current_chunk):
                        chunk_index += 1
                        # Start new chunk with overlap
                        overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                        current_chunk = overlap_text if overlap_text else ""
            elif len(current_chunk) >= min_chunk_size:
                # Paragraph doesn't fit, but current chunk is already substantial
                # Create chunk before adding the new paragraph
                if _create_chunk_if_ready(current_chunk):
                    chunk_index += 1
                    # Start new chunk with overlap
                    overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + para if overlap_text else para
            else:
                # Adding para would exceed chunk_size - create chunk first
                if current_chunk:
                    if _create_chunk_if_ready(current_chunk):
                        chunk_index += 1
                    
                    # Start new chunk with overlap from previous chunk
                    overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                else:
                    # No current chunk, but para doesn't fit - need to split para
                    current_chunk = para
            
            # If current chunk exceeds size after adding paragraph, split it by sentences
            if len(current_chunk) > chunk_size:
                # Split current chunk by sentences
                sentences = _split_text_by_separator(current_chunk, '. ')
                
                # Process sentences
                temp_chunk = ""
                for sentence in sentences:
                    # Add period back if it was removed
                    if not sentence.endswith('.') and '.' in current_chunk:
                        sentence += '.'
                    
                    # If sentence fits, add it
                    if len(temp_chunk) + len(sentence) + 2 <= chunk_size:
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                    else:
                        # Adding sentence would exceed chunk_size - create chunk first
                        if temp_chunk:
                            if _create_chunk_if_ready(temp_chunk):
                                chunk_index += 1
                            
                            # Start new temp_chunk with overlap
                            overlap_text = _get_overlap_text(temp_chunk, chunk_overlap)
                            temp_chunk = overlap_text + " " + sentence if overlap_text else sentence
                        else:
                            # Sentence itself is too large, split by characters
                            if len(sentence) > chunk_size:
                                # Split sentence into character chunks
                                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                                    char_chunk = sentence[i:i + chunk_size]
                                    if char_chunk and self.binary_validator.is_valid_text(char_chunk) and not self.binary_validator.contains_binary_patterns(char_chunk):
                                        chunks.append({
                                            "content": char_chunk,
                                            "page_number": page_number,
                                            "chunk_index": chunk_index,
                                            "metadata": {}
                                        })
                                        chunk_index += 1
                                temp_chunk = ""
                            else:
                                temp_chunk = sentence
                    
                    # Create chunk at natural boundary (end of sentence) if minimum size reached
                    if len(temp_chunk) >= min_chunk_size:
                        if _create_chunk_if_ready(temp_chunk):
                            chunk_index += 1
                            # Start new temp_chunk with overlap
                            overlap_text = _get_overlap_text(temp_chunk, chunk_overlap)
                            temp_chunk = overlap_text if overlap_text else ""
                    # Also create chunk if it reaches full chunk_size (standard behavior)
                    elif len(temp_chunk) >= chunk_size:
                        if _create_chunk_if_ready(temp_chunk):
                            chunk_index += 1
                            # Start new temp_chunk with overlap
                            overlap_text = _get_overlap_text(temp_chunk, chunk_overlap)
                            temp_chunk = overlap_text if overlap_text else ""
                
                # Update current_chunk with remaining temp_chunk
                current_chunk = temp_chunk
        
        # Add final chunk if it exists (even if below minimum size, to avoid losing content)
        if current_chunk:
            if self.binary_validator.is_valid_text(current_chunk) and not self.binary_validator.contains_binary_patterns(current_chunk):
                chunks.append({
                    "content": current_chunk,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "metadata": {}
                })
        
        return chunks
    
    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int):
        """
        Save text chunks to database with embeddings.
        
        DONE: Implemented chunk storage
        - Generate embeddings via VectorStore
        - Store in database with metadata
        - Validate chunks before saving (filter binary data)
        - Link related images/tables in metadata (future enhancement)
        
        Args:
            chunks: List of chunk dictionaries
            document_id: Document ID for database storage
        """
        saved_count = 0
        skipped_count = 0
        
        # Filter and validate chunks first
        valid_chunks = []
        for chunk in chunks:
            content = chunk.get("content", "")
            
            # Validate chunk content before saving - skip if binary data
            if not content or not content.strip():
                skipped_count += 1
                continue
            
            # Validate chunk content using BinaryValidator
            content_str = str(content)
            if not self.binary_validator.validate_chunk_content(content_str):
                print(f"Skipping chunk {chunk.get('chunk_index', '?')} on page {chunk.get('page_number', '?')} - validation failed")
                skipped_count += 1
                continue
            
            valid_chunks.append(chunk)
        
        # Process chunks in batches of 10 to avoid overwhelming the system
        # This allows concurrent embedding generation while preventing overload
        batch_size = 10
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            
            # Process batch concurrently (embeddings generated in parallel)
            tasks = []
            for chunk in batch:
                task = self.vector_store.store_chunk(
                    content=chunk.get("content", ""),
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    metadata=chunk.get("metadata", {})
                )
                tasks.append(task)
            
            # Wait for batch to complete (concurrent execution)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error saving chunk: {result}")
                    skipped_count += 1
                else:
                    saved_count += 1
        
        if skipped_count > 0:
            print(f"Saved {saved_count} chunks, skipped {skipped_count} chunks (binary data or invalid)")

