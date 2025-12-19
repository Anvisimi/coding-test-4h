"""
Document processing service using Docling

TODO: Implement this service to:
1. Parse PDF documents using Docling
2. Extract text, images, and tables
3. Store extracted content in database
4. Generate embeddings for text chunks
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.models.document import Document, DocumentChunk, DocumentImage, DocumentTable
from app.services.vector_store import VectorStore
from app.core.config import settings
import os
import time
import uuid
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from PIL import Image
import io
import json
import fitz  # PyMuPDF


class DocumentProcessor:
    """
    Process PDF documents and extract multimodal content.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStore(db)
        # Initialize DocumentConverter with PDF format
        # DocumentConverter expects allowed_formats as a list
        self.converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
    
    async def process_document(self, file_path: str, document_id: int) -> Dict[str, Any]:
        """
        Process a PDF document using Docling.
        
        Implementation steps:
        1. Update document status to 'processing'
        2. Use Docling to parse the PDF
        3. Extract and save text chunks
        4. Extract and save images
        5. Extract and save tables
        6. Generate embeddings for text chunks
        7. Update document status to 'completed'
        8. Handle errors appropriately
        
        Args:
            file_path: Path to the uploaded PDF file
            document_id: Database ID of the document
            
        Returns:
            {
                "status": "success" or "error",
                "text_chunks": <count>,
                "images": <count>,
                "tables": <count>,
                "processing_time": <seconds>
            }
        """
        start_time = time.time()
        
        try:
            # Update status to processing
            await self._update_document_status(document_id, "processing")
            
            # Parse PDF using Docling
            result = self.converter.convert(file_path)
            
            # Docling returns a ConversionResult object with .document attribute
            # Debug: Check what we actually got
            print(f"Result type: {type(result)}")
            print(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')][:15]}")
            
            # Get the document from result
            doc = result.document if hasattr(result, 'document') else result
            
            print(f"Doc type: {type(doc)}")
            print(f"Doc attributes: {[attr for attr in dir(doc) if not attr.startswith('_')][:20]}")
            
            # Extract and process content
            text_chunks_count = 0
            images_count = 0
            tables_count = 0
            
            # Get full document text - Docling's document has export_to_markdown() method
            full_text = ""
            try:
                if hasattr(doc, 'export_to_markdown'):
                    full_text = doc.export_to_markdown()
                    print(f"Got full text via export_to_markdown: {len(full_text)} chars")
            except Exception as e:
                print(f"Error in export_to_markdown: {e}")
            
            # Also try getting text directly
            if not full_text:
                try:
                    if hasattr(doc, 'text'):
                        full_text = doc.text
                        print(f"Got text directly: {len(full_text) if full_text else 0} chars")
                except Exception as e:
                    print(f"Error getting text: {e}")
            
            # Get pages - Docling document has .pages property
            pages = []
            try:
                if hasattr(doc, 'pages') and doc.pages:
                    pages = list(doc.pages)
                    print(f"Found {len(pages)} pages via .pages")
                elif hasattr(doc, 'items') and doc.items:
                    # Filter items that are pages
                    pages = [item for item in doc.items if hasattr(item, 'page') or hasattr(item, 'text')]
                    print(f"Found {len(pages)} pages via .items")
            except Exception as e:
                print(f"Error getting pages: {e}")
            
            # Get total pages
            total_pages = len(pages) if pages else 0
            if not total_pages and hasattr(doc, 'num_pages'):
                total_pages = doc.num_pages
            print(f"Total pages: {total_pages}")
            
            # Skip full text extraction entirely - the markdown export from Docling includes binary data
            # We'll rely on page-by-page extraction which is cleaner and more reliable
            # This prevents binary data from getting into the text chunks
            if full_text and full_text.strip():
                full_text_str = str(full_text).strip()
                
                # Quick check: if text is suspiciously long or contains any binary patterns, skip
                import re
                
                # Check for ANY corrupted hex patterns - even 1-2 occurrences is suspicious
                corrupted_patterns = [
                    r'x[0-9a-fA-F]{3,}',  # x9477, x834, etc.
                    r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b',  # ub0x0b, 5b0x0b pattern
                ]
                
                found_binary = False
                for pattern in corrupted_patterns:
                    matches = re.findall(pattern, full_text_str, re.IGNORECASE)
                    if len(matches) > 0:  # Even 1 match is suspicious
                        print(f"Skipping full text extraction - found {len(matches)} instances of corrupted hex pattern ({pattern})")
                        found_binary = True
                        break
                
                # Also check for Python byte strings and binary headers
                if not found_binary:
                    if "b'\\x" in full_text_str or 'b"\\x' in full_text_str:
                        print(f"Skipping full text extraction - contains Python byte string representations")
                        found_binary = True
                    elif any(header.lower() in full_text_str.lower() for header in ['\\x89png', '\\xff\\xd8\\xff', 'lhdr']):
                        print(f"Skipping full text extraction - contains binary header patterns")
                        found_binary = True
                
                if not found_binary:
                    # Sample a portion of the text to check for binary patterns
                    sample_text = full_text_str[:5000]  # Check first 5KB
                    if not self._is_valid_text(sample_text):
                        print(f"Skipping full text extraction - sample validation failed (first 200 chars: {repr(sample_text[:200])})")
                        found_binary = True
                
                if found_binary:
                    print("Skipping full text extraction - will rely on page-by-page extraction only (cleaner)")
                else:
                    # Only process if we're confident it's clean
                    # But still be conservative - if it's very long, might have hidden binary data
                    if len(full_text_str) > 20000:  # If > 20KB, be extra cautious
                        print(f"Skipping full text extraction - text is very long ({len(full_text_str)} chars), may contain hidden binary data")
                    else:
                        # Clean and validate
                        full_text_str = self._clean_markdown_text(full_text_str)
                        if self._is_valid_text(full_text_str):
                            chunks = self._chunk_text(full_text_str, document_id, 1)
                            if chunks:
                                await self._save_text_chunks(chunks, document_id)
                                text_chunks_count += len(chunks)
                                print(f"Created {len(chunks)} chunks from full text")
                        else:
                            print("Skipping full text extraction - validation failed after cleaning")
            
            # Extract images from document first (Docling stores pictures at document level)
            doc_pictures = []
            try:
                # Docling stores images as "pictures" in the document
                if hasattr(doc, 'pictures') and doc.pictures:
                    doc_pictures = list(doc.pictures) if not isinstance(doc.pictures, list) else doc.pictures
                    print(f"Found {len(doc_pictures)} pictures via doc.pictures")
                elif hasattr(doc, 'items') and doc.items:
                    # Extract picture items from document items
                    doc_pictures = [item for item in doc.items if hasattr(item, 'type') and 'picture' in str(getattr(item, 'type', '')).lower()]
                    print(f"Found {len(doc_pictures)} pictures via doc.items")
            except Exception as e:
                print(f"Error getting pictures from document: {e}")
            
            # Track if we successfully extracted any images from Docling
            successfully_extracted_from_docling = False
            
            # Use PyMuPDF to extract text from pages (more reliable than Docling page objects)
            # Open PDF with PyMuPDF for text extraction
            pdf_doc_for_text = None
            try:
                pdf_doc_for_text = fitz.open(file_path)
                print(f"Opened PDF with PyMuPDF for text extraction: {len(pdf_doc_for_text)} pages")
            except Exception as e:
                print(f"Error opening PDF with PyMuPDF for text extraction: {e}")
            
            # Process each page
            for page_idx, page in enumerate(pages, start=1):
                # Extract text and chunk it - try different ways to get text
                page_text = None
                page_num = page_idx
                
                # DEBUG: Log that we're processing this page
                if page_num <= 3:
                    print(f"DEBUG: Processing page {page_num} for text extraction")
                
                # Get page number from page object if available
                if hasattr(page, 'page'):
                    page_num = page.page
                elif hasattr(page, 'page_number'):
                    page_num = page.page_number
                
                # PRIMARY: Use PyMuPDF to extract text from the actual PDF page
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
                
                # FALLBACK: Try to get text from Docling page object
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
                    
                    # If no page text, try to extract from page's items
                    if not page_text and hasattr(page, 'items'):
                        page_text_parts = []
                        for item in page.items:
                            if hasattr(item, 'text') and item.text:
                                item_text = str(item.text)
                                # Validate each item before adding
                                if self._is_valid_text(item_text):
                                    page_text_parts.append(item_text)
                        if page_text_parts:
                            page_text = '\n\n'.join(page_text_parts)
                
                # DEBUG: Log if we found page text
                if page_num <= 3:
                    if page_text:
                        print(f"DEBUG: Found page {page_num} text ({len(str(page_text))} chars)")
                    else:
                        print(f"DEBUG: No text found for page {page_num}")
                
                if page_text:
                    # Convert to string and validate before processing
                    page_text_str = str(page_text).strip()
                    
                    # Skip empty text
                    if not page_text_str:
                        continue
                    
                    # DEBUG: Log first 200 chars to see what we're dealing with
                    if page_num <= 5:  # Only log first few pages to avoid spam
                        sample = repr(page_text_str[:200])
                        if "b'" in sample or 'b"' in sample or "\\x89" in sample:
                            print(f"DEBUG Page {page_num} text sample: {sample}")
                    
                    # Early validation - check for obvious binary patterns first
                    if page_text_str.startswith("b'") or page_text_str.startswith('b"'):
                        print(f"Skipping page {page_num} text - detected Python byte string representation")
                        continue
                    
                    # Check for binary patterns even if not at start - VERY AGGRESSIVE
                    # Look for the exact pattern we see: b'\x89Png or b'\r\n\x1a
                    if "b'\\x89" in page_text_str or 'b"\\x89' in page_text_str:
                        print(f"Skipping page {page_num} text - contains b'\\x89 pattern (PNG header in byte string)")
                        continue
                    if "b'\\r\\n\\x1a" in page_text_str or 'b"\\r\\n\\x1a' in page_text_str:
                        print(f"Skipping page {page_num} text - contains PNG header continuation pattern")
                        continue
                    
                    # Check for embedded binary patterns (b'\x89Png, b'\r\n, etc.)
                    # This catches binary data that's embedded in the middle of text
                    import re
                    should_skip_page = False
                    
                    # Check for ANY byte string pattern with hex escape sequences
                    # Pattern: b' or b" followed by \x and hex digits
                    byte_string_hex_pattern = r"b['\"].*?\\x[0-9a-fA-F]{2}"
                    matches = re.findall(byte_string_hex_pattern, page_text_str, re.IGNORECASE)
                    if len(matches) > 0:
                        # If we find ANY byte string with hex escapes, it's suspicious
                        # Count total occurrences
                        total_byte_strings = page_text_str.count("b'") + page_text_str.count('b"')
                        if total_byte_strings > 0:
                            # Check if any contain PNG/JPEG patterns
                            png_patterns = [r"\\x89Png", r"\\x89PNG", r"\\r\\n\\x1a", r"lhdr"]
                            for png_pattern in png_patterns:
                                if re.search(png_pattern, page_text_str, re.IGNORECASE):
                                    print(f"Skipping page {page_num} text - contains binary PNG/JPEG pattern ({png_pattern})")
                                    should_skip_page = True
                                    break
                            
                            # If we have multiple byte strings with hex escapes, it's binary
                            if not should_skip_page and len(matches) > 1:
                                print(f"Skipping page {page_num} text - contains {len(matches)} byte string patterns with hex escapes (likely binary data)")
                                should_skip_page = True
                    
                    # Also check for literal patterns (without regex)
                    if not should_skip_page:
                        binary_patterns = [
                            "b'\\x89Png", 'b"\\x89Png', "b'\\x89PNG", 'b"\\x89PNG',
                            "b'\\r\\n\\x1a", 'b"\\r\\n\\x1a',  # PNG header continuation
                            "b'\\x00\\x00\\x00", 'b"\\x00\\x00\\x00',  # Multiple null bytes
                        ]
                        for pattern in binary_patterns:
                            if pattern in page_text_str:
                                print(f"Skipping page {page_num} text - contains binary pattern: {pattern[:20]}...")
                                should_skip_page = True
                                break
                    
                    # Check for byte strings with escape sequences anywhere in text
                    if not should_skip_page:
                        if "b'\\x" in page_text_str or 'b"\\x' in page_text_str:
                            # Count occurrences - if ANY found, be very suspicious
                            byte_string_count = page_text_str.count("b'\\x") + page_text_str.count('b"\\x')
                            if byte_string_count > 0:
                                print(f"Skipping page {page_num} text - contains {byte_string_count} byte string patterns with hex escapes (likely binary data)")
                                should_skip_page = True
                    
                    if should_skip_page:
                        continue
                    
                    # Quick validation - skip if looks like binary data
                    if not self._is_valid_text(page_text_str):
                        print(f"Skipping page {page_num} text extraction - appears to be binary data (length: {len(page_text_str)}, first 100 chars: {repr(page_text_str[:100])})")
                        continue
                    
                    chunks = self._chunk_text(page_text_str, document_id, page_num)
                    if chunks:
                        # Final check: ensure no chunks contain binary data
                        clean_chunks = []
                        for chunk in chunks:
                            chunk_content = chunk.get("content", "")
                            # Check for binary patterns in chunk content
                            if "b'\\x89" in chunk_content or 'b"\\x89' in chunk_content:
                                print(f"Skipping chunk on page {page_num} - contains binary PNG pattern")
                                continue
                            if "b'\\r\\n\\x1a" in chunk_content or 'b"\\r\\n\\x1a' in chunk_content:
                                print(f"Skipping chunk on page {page_num} - contains PNG header continuation")
                                continue
                            clean_chunks.append(chunk)
                        
                        if clean_chunks:
                            await self._save_text_chunks(clean_chunks, document_id)
                            text_chunks_count += len(clean_chunks)
                            if page_num <= 5:
                                print(f"Created {len(clean_chunks)} chunks from page {page_num} text")
                
                # Extract images from page - try different structures
                page_images = []
                if hasattr(page, 'images') and page.images:
                    page_images = page.images if isinstance(page.images, list) else [page.images]
                elif hasattr(page, 'figures') and page.figures:
                    page_images = page.figures if isinstance(page.figures, list) else [page.figures]
                elif hasattr(page, 'pictures') and page.pictures:
                    page_images = page.pictures if isinstance(page.pictures, list) else [page.pictures]
                elif hasattr(page, 'items') and page.items:
                    # Extract picture items from page items
                    page_images = [item for item in page.items if hasattr(item, 'type') and 'picture' in str(getattr(item, 'type', '')).lower()]
                
                # Combine page images with document-level pictures on this page
                all_page_images = page_images.copy()
                for pic in doc_pictures:
                    pic_page = getattr(pic, 'page', None) or getattr(pic, 'page_number', None)
                    if pic_page == page_num or (pic_page is None and page_idx == 1):
                        all_page_images.append(pic)
                
                for img_idx, image_item in enumerate(all_page_images):
                    try:
                        # Debug: Check picture item structure
                        print(f"Picture item type: {type(image_item)}")
                        print(f"Picture attributes: {[attr for attr in dir(image_item) if not attr.startswith('_')][:15]}")
                        
                        # Try to get image data from Docling PictureItem
                        img_data = None
                        page_num_for_img = page_num
                        
                        # Get page number from image item if available
                        if hasattr(image_item, 'page'):
                            page_num_for_img = image_item.page
                        elif hasattr(image_item, 'page_number'):
                            page_num_for_img = image_item.page_number
                        elif hasattr(image_item, 'get_page'):
                            page_num_for_img = image_item.get_page()
                        
                        # Extract caption from Docling picture item
                        extracted_caption = None
                        # Try caption attribute (property)
                        if hasattr(image_item, 'caption'):
                            try:
                                caption_val = image_item.caption
                                # Check if it's a method (callable) or a property
                                if callable(caption_val):
                                    extracted_caption = str(caption_val())
                                elif caption_val:
                                    extracted_caption = str(caption_val)
                            except Exception as e:
                                print(f"Error extracting Docling caption attribute: {e}")
                        
                        # Try title attribute (property)
                        if not extracted_caption and hasattr(image_item, 'title'):
                            try:
                                title_val = image_item.title
                                # Check if it's a method (callable) or a property
                                if callable(title_val):
                                    extracted_caption = str(title_val())
                                elif title_val:
                                    extracted_caption = str(title_val)
                            except Exception as e:
                                print(f"Error extracting Docling title attribute: {e}")
                        
                        # Try get_caption method
                        if not extracted_caption and hasattr(image_item, 'get_caption'):
                            try:
                                caption_result = image_item.get_caption()
                                if caption_result:
                                    extracted_caption = str(caption_result)
                            except Exception as e:
                                print(f"Error calling Docling get_caption: {e}")
                        
                        # Try get_title method
                        if not extracted_caption and hasattr(image_item, 'get_title'):
                            try:
                                title_result = image_item.get_title()
                                if title_result:
                                    extracted_caption = str(title_result)
                            except Exception as e:
                                print(f"Error calling Docling get_title: {e}")
                        
                        # Docling PictureItem.get_image() requires the document as argument
                        if hasattr(image_item, 'get_image'):
                            try:
                                img_result = image_item.get_image(doc)
                                print(f"get_image(doc) returned: {type(img_result)}")
                                # Check what type of object is returned
                                if img_result:
                                    if isinstance(img_result, Image.Image):
                                        img_data = img_result
                                        print(f"Got PIL Image via get_image(doc)")
                                    elif isinstance(img_result, bytes):
                                        img_data = img_result
                                        print(f"Got bytes via get_image(doc)")
                                    elif hasattr(img_result, 'image'):
                                        img_data = img_result.image
                                        print(f"Got image from result.image")
                                    elif hasattr(img_result, 'data'):
                                        img_data = img_result.data
                                        print(f"Got image from result.data")
                                    else:
                                        # Try to convert to PIL Image
                                        try:
                                            img_data = Image.open(io.BytesIO(img_result))
                                            print(f"Converted get_image(doc) result to PIL Image")
                                        except:
                                            print(f"Could not convert get_image(doc) result: {type(img_result)}")
                            except Exception as e:
                                print(f"Error in get_image(doc): {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Try export_to_image() if available
                        if not img_data and hasattr(image_item, 'export_to_image'):
                            try:
                                img_data = image_item.export_to_image()
                                print(f"Got image via export_to_image")
                            except Exception as e:
                                print(f"Error in export_to_image: {e}")
                        
                        # Try accessing image property (might be None, but check)
                        if not img_data and hasattr(image_item, 'image'):
                            img_data = image_item.image
                            if img_data:
                                print(f"Got image via .image property")
                            else:
                                print(f".image property is None or empty")
                        
                        # Try data property
                        if not img_data and hasattr(image_item, 'data'):
                            img_data = image_item.data
                            print(f"Got image via .data property")
                        
                        # Try content property
                        if not img_data and hasattr(image_item, 'content'):
                            img_data = image_item.content
                            print(f"Got image via .content property")
                        
                        # Try bytes property
                        if not img_data and hasattr(image_item, 'bytes'):
                            img_data = image_item.bytes
                            print(f"Got image via .bytes property")
                        
                        # Try render() method
                        if not img_data and hasattr(image_item, 'render'):
                            try:
                                img_data = image_item.render()
                                print(f"Got image via render()")
                            except Exception as e:
                                print(f"Error in render(): {e}")
                        
                        # If it's already a PIL Image or bytes
                        if isinstance(image_item, Image.Image):
                            img_data = image_item
                            print(f"Image item is already PIL Image")
                        elif isinstance(image_item, bytes):
                            img_data = image_item
                            print(f"Image item is already bytes")
                        
                        if img_data:
                            await self._save_image(
                                img_data,
                                document_id,
                                page_num_for_img,
                                {"index": img_idx, "source": "docling"},
                                caption=extracted_caption
                            )
                            images_count += 1
                            successfully_extracted_from_docling = True
                            print(f"Saved image {img_idx} from page {page_num_for_img} with caption: {extracted_caption}")
                        else:
                            print(f"Could not extract image data from picture item {img_idx}")
                    except Exception as e:
                        print(f"Error saving image on page {page_num}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # First, identify which pages have tables (to avoid rendering them as images)
            pages_with_tables = set()
            try:
                if hasattr(doc, 'tables') and doc.tables:
                    tables_preview = list(doc.tables) if not isinstance(doc.tables, list) else doc.tables
                    for table_item in tables_preview:
                        page_num = getattr(table_item, 'page', None) or getattr(table_item, 'page_number', None)
                        if page_num:
                            pages_with_tables.add(page_num)
                    print(f"Pages with tables (from Docling): {sorted(pages_with_tables)}")
                
                # Also check pages using PyMuPDF to detect table-like structures
                # This helps catch tables that Docling might have missed or pages with table-like content
                try:
                    pdf_doc_temp = fitz.open(file_path)
                    for page_num in range(len(pdf_doc_temp)):
                        page = pdf_doc_temp[page_num]
                        page_number = page_num + 1
                        
                        # Check if page has table-like structures using PyMuPDF
                        # Look for blocks that might be tables
                        blocks = page.get_text("blocks")
                        table_indicators = 0
                        
                        # Check for table-like patterns in text blocks
                        for block in blocks:
                            block_text = block[4] if len(block) > 4 else ""
                            # Look for patterns that suggest tables (multiple spaces, tabs, pipe characters)
                            if block_text:
                                # Count pipe characters (common in markdown tables)
                                if block_text.count('|') > 5:
                                    table_indicators += 1
                                # Check for multiple consecutive spaces (table-like alignment)
                                if '  ' in block_text and block_text.count('  ') > 3:
                                    table_indicators += 1
                                # Check for tab characters
                                if '\t' in block_text:
                                    table_indicators += 1
                        
                        # If we found multiple table indicators, mark this page
                        if table_indicators >= 2:
                            pages_with_tables.add(page_number)
                            print(f"Page {page_number} marked as table page (PyMuPDF detection: {table_indicators} indicators)")
                    
                    pdf_doc_temp.close()
                    print(f"Total pages with tables (combined): {sorted(pages_with_tables)}")
                except Exception as e:
                    print(f"Error in PyMuPDF table detection: {e}")
                    
            except Exception as e:
                print(f"Error identifying pages with tables: {e}")
            
            # Always use PyMuPDF to extract ALL images (embedded + vector graphics)
            # This supplements Docling extraction which may miss some images
            try:
                print("Extracting images via PyMuPDF (embedded + vector graphics)...")
                pdf_doc = fitz.open(file_path)
                pymupdf_image_count = 0
                seen_image_xrefs = set()  # Track extracted images to avoid duplicates
                
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    page_number = page_num + 1
                    
                    # Skip rendering full pages that contain tables (tables are extracted separately)
                    is_table_page = page_number in pages_with_tables
                    
                    # Method 1: Extract embedded raster images
                    image_list = page.get_images(full=True)  # full=True gets more image info
                    print(f"Page {page_number}: Found {len(image_list)} embedded images")
                    
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            # Skip if we've already extracted this image
                            if xref in seen_image_xrefs:
                                continue
                            seen_image_xrefs.add(xref)
                            
                            base_image = pdf_doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Try to extract caption from PDF text near the image
                            caption = None
                            try:
                                # Get image bounding box
                                img_bbox = page.get_image_bbox(img)
                                if img_bbox:
                                    caption = self._extract_caption_from_page(page, img_bbox)
                            except Exception as e:
                                print(f"Error extracting caption for image {img_idx}: {e}")
                            
                            # Save image directly using PyMuPDF extracted data
                            await self._save_image(
                                image_bytes,
                                document_id,
                                page_number,
                                {"index": img_idx, "source": "pymupdf_embedded", "xref": xref},
                                caption=caption
                            )
                            pymupdf_image_count += 1
                            print(f"Saved embedded image {img_idx} from page {page_number} via PyMuPDF with caption: {caption}")
                        except Exception as e:
                            print(f"Error extracting embedded image from PDF page {page_number}: {e}")
                            continue
                    
                    # Method 2: Extract images by rendering their bounding boxes
                    # This captures images that might be vector graphics or complex figures
                    # Skip if page is text-heavy (indicates full page with text, not a figure)
                    try:
                        page_text = page.get_text()
                        text_length = len(page_text.strip()) if page_text else 0
                        is_text_heavy = text_length > 1000  # Skip pages with lots of text
                        
                        if not is_text_heavy:
                            # Get images again with full=True to get bounding boxes
                            image_list_full = page.get_images(full=True)
                            rendered_image_xrefs = set()
                            
                            for img_idx, img in enumerate(image_list_full):
                                try:
                                    xref = img[0]
                                    # Skip if we already extracted this as embedded image
                                    if xref in seen_image_xrefs:
                                        continue
                                    
                                    # Get bounding box for this image
                                    try:
                                        bbox = page.get_image_bbox(img)
                                        if bbox and bbox.width > 50 and bbox.height > 50:  # Only extract if reasonably sized
                                            # Render the image region at high resolution
                                            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                                            
                                            # Convert pixmap to PIL Image
                                            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            
                                            # Try to extract caption from text near the image
                                            caption = None
                                            try:
                                                caption = self._extract_caption_from_page(page, bbox)
                                            except:
                                                pass
                                            
                                            # Save the rendered image
                                            await self._save_image(
                                                img_data,
                                                document_id,
                                                page_number,
                                                {"index": img_idx, "source": "pymupdf_bbox", "xref": xref, "bbox": str(bbox)},
                                                caption=caption
                                            )
                                            pymupdf_image_count += 1
                                            rendered_image_xrefs.add(xref)
                                            print(f"Saved image from bbox {img_idx} on page {page_number} via PyMuPDF with caption: {caption}")
                                            
                                            pix = None  # Free memory
                                    except Exception as e:
                                        # get_image_bbox might fail for some images, skip them
                                        print(f"Could not get bbox for image {xref} on page {page_number}: {e}")
                                        continue
                                except Exception as e:
                                    print(f"Error processing image bbox on page {page_number}: {e}")
                                    continue
                        else:
                            print(f"Skipping bbox extraction for page {page_number} (text-heavy page with {text_length} chars)")
                    except Exception as e:
                        print(f"Error checking page text for bbox extraction: {e}")
                    
                    # Method 3: Extract individual figure regions from pages with vector graphics
                    # Extract actual figure regions, not full pages with text/tables
                    drawings = page.get_drawings()
                    if drawings and not is_table_page:
                        print(f"Page {page_number}: Found {len(drawings)} vector graphics")
                        
                        try:
                            page_text = page.get_text()
                            text_length = len(page_text.strip()) if page_text else 0
                            
                            # Try to extract individual figure regions by finding drawing clusters
                            # Group drawings by proximity to identify figure regions
                            if len(drawings) > 5:  # Only process pages with significant vector graphics
                                # Calculate bounding box of all drawings to find figure regions
                                drawing_rects = []
                                for drawing in drawings:
                                    if 'rect' in drawing:
                                        drawing_rects.append(drawing['rect'])
                                
                                if drawing_rects:
                                    # Find the overall bounding box of all drawings
                                    min_x = min(r.x0 for r in drawing_rects)
                                    min_y = min(r.y0 for r in drawing_rects)
                                    max_x = max(r.x1 for r in drawing_rects)
                                    max_y = max(r.y1 for r in drawing_rects)
                                    
                                    # Create a bounding box for the figure region
                                    figure_bbox = fitz.Rect(min_x, min_y, max_x, max_y)
                                    
                                    # Only extract if the figure region is reasonably sized
                                    # and doesn't cover the entire page (which would indicate text page)
                                    page_rect = page.rect
                                    figure_area_ratio = (figure_bbox.width * figure_bbox.height) / (page_rect.width * page_rect.height)
                                    
                                    # Extract figure region if:
                                    # 1. Figure region is substantial (>10% of page but <80% to avoid full-page)
                                    # 2. Page is not text-heavy (skip if >1000 chars of text - indicates full page with text)
                                    # 3. Or if it's a figure-only page (many drawings, little text)
                                    is_figure_only_page = (len(drawings) > 20 and text_length < 500)
                                    is_text_heavy = text_length > 1000  # Skip pages with lots of text
                                    is_substantial_figure = ((0.1 < figure_area_ratio < 0.8) or is_figure_only_page) and not is_text_heavy
                                    
                                    if is_substantial_figure:
                                        try:
                                            # Render the figure region at high resolution
                                            mat = fitz.Matrix(2.0, 2.0)
                                            # Add some padding around the figure
                                            padding = 20
                                            clip_rect = fitz.Rect(
                                                max(0, figure_bbox.x0 - padding),
                                                max(0, figure_bbox.y0 - padding),
                                                min(page_rect.width, figure_bbox.x1 + padding),
                                                min(page_rect.height, figure_bbox.y1 + padding)
                                            )
                                            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                                            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                            
                                            # Try to extract caption from PDF text near the figure
                                            caption = None
                                            try:
                                                caption = self._extract_caption_from_page(page, clip_rect)
                                            except:
                                                pass
                                            
                                            await self._save_image(
                                                img_data,
                                                document_id,
                                                page_number,
                                                {"index": 0, "source": "pymupdf_figure_region", "type": "vector_graphics", 
                                                 "drawings": len(drawings), "text_chars": text_length, "bbox": str(clip_rect)},
                                                caption=caption
                                            )
                                            pymupdf_image_count += 1
                                            print(f"Saved figure region from page {page_number} (area_ratio={figure_area_ratio:.2f}, {len(drawings)} drawings, {text_length} chars text)")
                                            
                                            pix = None
                                        except Exception as e:
                                            print(f"Error rendering figure region from page {page_number}: {e}")
                                    else:
                                        # If figure region is too large (covers most of page), it might be a text page
                                        # Only render if it's clearly a figure-only page
                                        if is_figure_only_page:
                                            try:
                                                mat = fitz.Matrix(2.0, 2.0)
                                                pix = page.get_pixmap(matrix=mat)
                                                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                                
                                                await self._save_image(
                                                    img_data,
                                                    document_id,
                                                    page_number,
                                                    {"index": 0, "source": "pymupdf_vector", "type": "vector_graphics_figure", 
                                                     "drawings": len(drawings), "text_chars": text_length}
                                                )
                                                pymupdf_image_count += 1
                                                print(f"Saved rendered page {page_number} as image (figure-only page: {len(drawings)} vector graphics, {text_length} chars text)")
                                                
                                                pix = None
                                            except Exception as e:
                                                print(f"Error rendering page {page_number} with vector graphics: {e}")
                                        else:
                                            if is_text_heavy:
                                                print(f"Skipping page {page_number} (text-heavy page with {text_length} chars - likely full page with text)")
                                            else:
                                                print(f"Skipping page {page_number} (figure region too large: area_ratio={figure_area_ratio:.2f}, {text_length} chars)")
                        except Exception as e:
                            print(f"Error analyzing page {page_number} characteristics: {e}")
                            import traceback
                            traceback.print_exc()
                    elif is_table_page:
                        print(f"Skipping vector graphics extraction for page {page_number} (contains tables - extracted separately)")
                
                # Combine counts: use the maximum to avoid double-counting
                # If PyMuPDF found more, use that count; otherwise add to existing
                if pymupdf_image_count > 0:
                    # If Docling didn't extract successfully, use PyMuPDF count
                    if not successfully_extracted_from_docling:
                        images_count = pymupdf_image_count
                    else:
                        # Use the maximum to avoid counting the same image twice
                        images_count = max(images_count, pymupdf_image_count)
                    
                    print(f"Total images extracted: {images_count} (PyMuPDF contributed {pymupdf_image_count})")
                
                pdf_doc.close()
            except Exception as e:
                print(f"Error in PyMuPDF extraction: {e}")
                import traceback
                traceback.print_exc()
            
            # Extract tables from document - Docling document has .tables property
            tables = []
            try:
                if hasattr(doc, 'tables') and doc.tables:
                    tables = list(doc.tables) if not isinstance(doc.tables, list) else doc.tables
                    print(f"Found {len(tables)} tables via .tables")
                elif hasattr(result, 'tables') and result.tables:
                    tables = list(result.tables) if not isinstance(result.tables, list) else result.tables
                    print(f"Found {len(tables)} tables via result.tables")
                elif hasattr(doc, 'items') and doc.items:
                    # Extract tables from items
                    tables = [item for item in doc.items if hasattr(item, 'type') and str(getattr(item, 'type', '')).lower() == 'table']
                    print(f"Found {len(tables)} tables via .items")
            except Exception as e:
                print(f"Error getting tables: {e}")
            
            print(f"Total tables found: {len(tables)}")
            
            if tables:
                for table_idx, table_item in enumerate(tables):
                    try:
                        page_num = getattr(table_item, 'page', 1) if hasattr(table_item, 'page') else 1
                        await self._save_table(
                            table_item,
                            document_id,
                            page_num,
                            {"index": table_idx, "source": "docling"}
                        )
                        tables_count += 1
                    except Exception as e:
                        print(f"Error saving table {table_idx}: {e}")
                        continue
            
            # Update document with counts
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.total_pages = total_pages
                document.text_chunks_count = text_chunks_count
                document.images_count = images_count
                document.tables_count = tables_count
                self.db.commit()
            
            # Close PyMuPDF document if opened
            if pdf_doc_for_text:
                try:
                    pdf_doc_for_text.close()
                except:
                    pass
            
            # Update status to completed
            processing_time = time.time() - start_time
            await self._update_document_status(document_id, "completed")
            
            return {
                "status": "success",
                "text_chunks": text_chunks_count,
                "images": images_count,
                "tables": tables_count,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            error_msg = str(e)
            await self._update_document_status(document_id, "error", error_msg)
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "text_chunks": 0,
                "images": 0,
                "tables": 0,
                "processing_time": round(processing_time, 2),
                "error": error_msg
            }
    
    def _clean_markdown_text(self, text: str) -> str:
        """
        Clean markdown text by removing binary data sections.
        Splits text into lines and filters out lines that look like binary data.
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Skip lines that look like binary data
            # Check for Python byte string representations (b'... or b"...)
            if line_stripped.startswith("b'") or line_stripped.startswith('b"'):
                removed_count += 1
                continue
            
            # Check if line contains byte string patterns anywhere (not just at start)
            if "b'\\x" in line_stripped or 'b"\\x' in line_stripped:
                removed_count += 1
                continue
            
            # Check for excessive escape sequences in this line
            escape_count = line_stripped.count('\\x')
            if escape_count > 5:  # More than 5 escape sequences in one line (reduced from 10)
                removed_count += 1
                continue
            
            # Check for binary patterns (PNG, JPEG headers)
            line_lower = line_stripped.lower()
            if '\\x89png' in line_lower or '\\x89png' in line_lower:
                removed_count += 1
                continue
            if '\\xff\\xd8\\xff' in line_lower:  # JPEG header
                removed_count += 1
                continue
            if 'lhdr' in line_lower and escape_count > 2:  # PNG chunk header with escape sequences
                removed_count += 1
                continue
            
            # Check if line is mostly non-printable
            printable_count = sum(1 for c in line_stripped if c.isprintable() or c.isspace())
            if len(line_stripped) > 0 and printable_count / len(line_stripped) < 0.6:  # Stricter threshold
                removed_count += 1
                continue
            
            # Check for lines that are mostly hex/escape sequences
            hex_like_chars = sum(1 for c in line_stripped if c in '0123456789abcdefABCDEF\\x')
            if len(line_stripped) > 20 and hex_like_chars / len(line_stripped) > 0.4:  # More than 40% hex-like
                removed_count += 1
                continue
            
            # Line looks okay, keep it
            cleaned_lines.append(line)
        
        if removed_count > 0:
            print(f"Cleaned markdown: removed {removed_count} lines containing binary data")
        
        return '\n'.join(cleaned_lines)
    
    def _is_valid_text(self, text: str, min_printable_ratio: float = 0.8) -> bool:
        """
        Check if text is valid readable text (not binary data).
        
        Args:
            text: Text to validate
            min_printable_ratio: Minimum ratio of printable characters (default 0.8)
        
        Returns:
            True if text appears to be valid readable text
        """
        if not text or len(text) == 0:
            return False
        
        # Check length - if too long, likely binary data
        if len(text) > 50000:  # 50KB limit (reduced from 100KB)
            return False
        
        # Check for Python byte string representations (b'...', b"...")
        # Check both at start and anywhere in text (might be embedded)
        text_stripped = text.strip()
        if text_stripped.startswith("b'") or text_stripped.startswith('b"'):
            return False
        
        # Check if text contains Python byte string representations anywhere
        # Look for patterns like b'\x89PNG', b'\x89Png, b'\r\n, etc.
        # ANY occurrence of b'\x or b"\x is suspicious
        if "b'\\x" in text or 'b"\\x' in text:
            return False
        
        # Check for byte strings with escape sequences (b'\r, b'\n, b'\x00, etc.)
        byte_string_patterns = ["b'\\r", 'b"\\r', "b'\\n", 'b"\\n', "b'\\x00", 'b"\\x00', "b'\\x89", 'b"\\x89']
        for pattern in byte_string_patterns:
            if pattern in text:
                return False
        
        # Check for byte string patterns like b'\x89Png (common PNG header in byte strings)
        # Check multiple variations and case-insensitive
        png_byte_patterns = [
            "b'\\x89Png", 'b"\\x89Png', "b'\\x89PNG", 'b"\\x89PNG',
            "b'\\x89png", 'b"\\x89png',  # lowercase variant
            "b'\\x89Png\\r", 'b"\\x89Png\\r',  # with carriage return
            "b'\\x89Png\\r\\n", 'b"\\x89Png\\r\\n',  # with CRLF
            "b'\\x89Png\\r\\n\\x1a", 'b"\\x89Png\\r\\n\\x1a',  # PNG header continuation
            "b'\\x89Png\\r\\n\\x1a\\n", 'b"\\x89Png\\r\\n\\x1a\\n',  # Full PNG header start
        ]
        for pattern in png_byte_patterns:
            if pattern in text:
                return False
        
        # Also check using regex for more flexible matching
        import re
        png_regex_patterns = [
            r"b['\"]\\x89Png",  # Case-insensitive PNG header
            r"b['\"]\\r\\n\\x1a\\n",  # PNG header continuation
            r"b['\"]\\x00\\x00\\x00",  # Multiple null bytes (common in binary)
        ]
        for pattern in png_regex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Check for byte strings with lhdr (PNG chunk header)
        # If text contains b' or b" AND lhdr, it's likely binary
        if ("b'" in text or 'b"' in text) and 'lhdr' in text.lower():
            return False
        
        # More aggressive: if text contains b' or b" followed by escape sequences, it's binary
        # Check for patterns like b'\x... anywhere in text
        import re
        byte_string_with_escape = re.search(r"b['\"].*?\\x[0-9a-fA-F]{2}", text)
        if byte_string_with_escape:
            # Count how many escape sequences follow b' or b"
            matches = re.findall(r"b['\"].*?\\x[0-9a-fA-F]{2}", text)
            if len(matches) > 2:  # More than 2 occurrences is definitely binary
                return False
            # Even 1-2 occurrences with PNG-like patterns is suspicious
            if any('png' in match.lower() or 'lhdr' in match.lower() for match in matches):
                return False
        
        # Check for common binary data patterns in string form
        binary_string_patterns = [
            "\\x89PNG",  # PNG header as string
            "\\x89Png",  # PNG header (case variant)
            "\\xff\\xd8\\xff",  # JPEG header as string
            "lhdr",  # PNG chunk header (but only if combined with escape sequences)
            "\\x00\\x00\\x00",  # Multiple null bytes as string
            "\\r\\n\\x1a\\n",  # PNG header continuation
        ]
        text_lower = text.lower()
        binary_pattern_count = 0
        for pattern in binary_string_patterns:
            count = text_lower.count(pattern.lower())
            if count > 0:
                binary_pattern_count += count
                # If we find PNG/JPEG headers, it's definitely binary
                if "png" in pattern.lower() or "jpeg" in pattern.lower() or "\\xff\\xd8" in pattern.lower():
                    if count >= 1:
                        return False
        
        # If we have many binary patterns, it's likely binary data
        if binary_pattern_count > 3:
            return False
        
        # Count printable characters
        printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
        printable_ratio = printable_count / len(text) if len(text) > 0 else 0
        
        # Check for binary patterns in actual bytes (PNG, JPEG headers, etc.)
        try:
            text_bytes = text.encode('utf-8', errors='ignore')[:200]  # Check first 200 bytes
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
        
        # Check for excessive non-printable characters
        if printable_ratio < min_printable_ratio:
            return False
        
        # Check for excessive escape sequences (like \x89, \xa4, etc.)
        # Lower threshold - even 2% is suspicious for text content
        escape_seq_count = text.count('\\x')
        if escape_seq_count > 0:
            escape_ratio = escape_seq_count / len(text) if len(text) > 0 else 0
            if escape_ratio > 0.02:  # More than 2% escape sequences (very strict)
                return False
            # If we have many escape sequences, it's likely binary
            if escape_seq_count > 20:  # Reduced from 50 - even 20 is suspicious
                return False
        
        # Check for patterns like b'\x89Png or b'\r\n\x1a\n (PNG headers in byte strings)
        # These are strong indicators of binary data, even if ratio is low
        if "b'\\x89" in text or 'b"\\x89' in text:
            # If we find PNG header patterns, it's definitely binary
            return False
        if "b'\\r\\n\\x1a" in text or 'b"\\r\\n\\x1a' in text:
            # PNG header continuation
            return False
        
        # Check for excessive null bytes
        null_byte_count = text.count('\x00')
        if null_byte_count > 0:
            null_ratio = null_byte_count / len(text) if len(text) > 0 else 0
            if null_ratio > 0.02:  # More than 2% null bytes (reduced from 5%)
                return False
            # If we have many null bytes, it's likely binary
            if null_byte_count > 20:
                return False
        
        # Check for excessive non-ASCII characters that aren't common in text
        # Count bytes that are outside normal ASCII printable range
        non_ascii_count = sum(1 for c in text if ord(c) > 127 and not c.isprintable())
        if non_ascii_count > len(text) * 0.1:  # More than 10% non-printable non-ASCII
            return False
        
        # Check if text looks like it's mostly hex/escape sequences
        # If more than 30% of characters are part of escape sequences or hex, it's likely binary
        hex_like_chars = sum(1 for c in text if c in '0123456789abcdefABCDEF\\x')
        if len(text) > 100 and hex_like_chars / len(text) > 0.3:
            # But allow if it's clearly readable text with some hex
            if escape_seq_count / len(text) > 0.02:  # If escape sequences are significant
                return False
        
        # Check for corrupted escape sequences (like "x10", "x42", "x9477" without backslash)
        # Pattern: letter 'x' followed by hex digits, repeated many times
        import re
        # Match patterns like x9477, x834, xa6c4, etc. (corrupted Unicode escapes)
        corrupted_hex_pattern = re.compile(r'x[0-9a-fA-F]{2,}')
        corrupted_hex_matches = corrupted_hex_pattern.findall(text)
        if len(corrupted_hex_matches) > 5:  # More than 5 occurrences is suspicious
            # Check if it's a significant portion of the text
            total_corrupted_length = sum(len(m) for m in corrupted_hex_matches)
            if len(text) > 100 and total_corrupted_length / len(text) > 0.05:  # More than 5% corrupted hex
                return False
            # Also check if we have many consecutive corrupted hex sequences
            if len(corrupted_hex_matches) > 20:  # Many occurrences
                return False
        
        # Check for patterns like "x10421", "x1xx", "x51xx" - corrupted binary data
        corrupted_binary_pattern = re.compile(r'x[0-9a-fA-F]{2,}x{2,}')
        if corrupted_binary_pattern.search(text):
            return False
        
        # Check for sequences of corrupted hex like "x9477x834xa6c4" (common pattern)
        consecutive_corrupted = re.compile(r'x[0-9a-fA-F]{2,}(?:x[0-9a-fA-F]{2,}){3,}')
        if consecutive_corrupted.search(text):
            return False
        
        # Check for patterns like "ub0x0b", "5b0x0b" - corrupted Unicode escape sequences
        # Pattern: letter/digit + hex + "x0x0b"
        corrupted_unicode_pattern = re.compile(r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b', re.IGNORECASE)
        unicode_corrupted_matches = corrupted_unicode_pattern.findall(text)
        if len(unicode_corrupted_matches) > 2:  # More than 2 occurrences is suspicious
            return False
        
        # Check for repeated patterns like "ub0x0bub0x0b" or similar
        repeated_corrupted = re.compile(r'([a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b){2,}', re.IGNORECASE)
        if repeated_corrupted.search(text):
            return False
        
        return True
    
    def _chunk_text(self, text: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks for vector storage.
        
        TODO: Implement text chunking strategy
        - Split by sentences or paragraphs
        - Maintain context with overlap
        - Keep metadata (page number, position, etc.)
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or not text.strip():
            return []
        
        # Validate text - skip if it's binary data
        if not self._is_valid_text(text):
            print(f"Skipping invalid text on page {page_number} (likely binary data, length: {len(text)})")
            return []
        
        # Limit text length to prevent extremely long extractions
        max_text_length = 50000  # 50KB limit per page
        if len(text) > max_text_length:
            print(f"Truncating text on page {page_number} from {len(text)} to {max_text_length} characters")
            text = text[:max_text_length]
        
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        # Enforce maximum chunk size (prevent extremely long chunks)
        max_chunk_size = 2000  # Maximum 2000 characters per chunk
        if chunk_size > max_chunk_size:
            chunk_size = max_chunk_size
        
        # Split by paragraphs first, then by sentences if needed
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if it exists and is valid
                # Validate chunk before adding - double check for binary patterns
                if current_chunk and self._is_valid_text(current_chunk):
                    # Additional check: ensure no binary patterns slipped through
                    if not ("b'\\x" in current_chunk or 'b"\\x' in current_chunk):
                        chunks.append({
                            "content": current_chunk,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            "metadata": {}
                        })
                        chunk_index += 1
                    else:
                        print(f"Skipping chunk on page {page_number} - contains binary byte string pattern")
                
                # If paragraph itself is too large, split by sentences
                if len(para) > chunk_size:
                    sentences = para.split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                            if current_chunk:
                                current_chunk += ". " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            # Validate chunk before adding
                            chunk_to_add = current_chunk + "."
                            if chunk_to_add and self._is_valid_text(chunk_to_add):
                                # Additional check: ensure no binary patterns slipped through
                                if not ("b'\\x" in chunk_to_add or 'b"\\x' in chunk_to_add):
                                    chunks.append({
                                        "content": chunk_to_add,
                                        "page_number": page_number,
                                        "chunk_index": chunk_index,
                                        "metadata": {}
                                    })
                                    chunk_index += 1
                                else:
                                    print(f"Skipping chunk on page {page_number} - contains binary byte string pattern")
                            
                            # Start new chunk with overlap
                            if chunk_overlap > 0 and chunks:
                                # Get last few words from previous chunk for overlap
                                prev_words = chunks[-1]["content"].split()[-chunk_overlap//10:]
                                current_chunk = " ".join(prev_words) + " " + sentence
                            else:
                                current_chunk = sentence
                    
                    # Add remaining chunk - validate it first
                    if current_chunk:
                        current_chunk += "." if not current_chunk.endswith(".") else ""
                        # Validate before continuing - skip if binary
                        if not self._is_valid_text(current_chunk) or ("b'\\x" in current_chunk or 'b"\\x' in current_chunk):
                            print(f"Skipping remaining chunk on page {page_number} - contains binary data")
                            current_chunk = ""  # Clear it so it doesn't get added later
                else:
                    # Start new chunk with overlap
                    if chunk_overlap > 0 and chunks:
                        prev_words = chunks[-1]["content"].split()[-chunk_overlap//10:]
                        current_chunk = " ".join(prev_words) + "\n\n" + para
                    else:
                        current_chunk = para
        
        # Add final chunk
        if current_chunk:
            # Validate final chunk before adding
            if self._is_valid_text(current_chunk):
                # Additional check: ensure no binary patterns slipped through
                if not ("b'\\x" in current_chunk or 'b"\\x' in current_chunk):
                    chunks.append({
                        "content": current_chunk,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "metadata": {}
                    })
                else:
                    print(f"Skipping final chunk on page {page_number} - contains binary byte string pattern")
        
        return chunks
    
    def _extract_caption_from_page(self, page, bbox) -> str:
        """
        Extract caption text from PDF page near the given bounding box.
        Looks for captions above, below, and near the image/figure.
        Returns full multi-line captions without truncation.
        """
        try:
            import fitz
            text_blocks = page.get_text("blocks")
            if not text_blocks:
                return None
            
            # Look for caption patterns
            caption_patterns = ["Figure", "Fig.", "Fig ", "Figure ", "FIGURE", "FIG."]
            
            # Search area: above the image (within 150 pixels) and below (within 100 pixels)
            # Increased to capture captions that might be further away
            search_above_y = bbox.y0 + 150
            search_below_y = bbox.y1 + 100
            horizontal_tolerance = 200  # Pixels - increased to capture wider captions
            
            caption_blocks = []
            found_caption_start = False
            
            # Collect all potential caption blocks first, then filter
            potential_blocks = []
            
            for block in text_blocks:
                block_rect = fitz.Rect(block[:4])
                block_text = block[4] if len(block) > 4 else ""
                
                if not block_text:
                    continue
                
                # Check if block is near the image horizontally
                block_center_x = (block_rect.x0 + block_rect.x1) / 2
                bbox_center_x = (bbox.x0 + bbox.x1) / 2
                horizontal_distance = abs(block_center_x - bbox_center_x)
                
                # Check if block is above or below the image
                is_above = block_rect.y1 < search_above_y and block_rect.y1 > bbox.y0 - 200
                is_below = block_rect.y0 > bbox.y1 and block_rect.y0 < search_below_y
                
                if horizontal_distance > horizontal_tolerance:
                    continue
                
                if is_above or is_below:
                    # Store potential caption blocks with their position
                    block_text_upper = block_text.upper()
                    has_pattern = any(pattern.upper() in block_text_upper for pattern in caption_patterns)
                    potential_blocks.append((block_rect.y0, block_text, has_pattern))
            
            # Sort by vertical position
            potential_blocks.sort(key=lambda x: x[0])
            
            # Find the first block with a caption pattern
            caption_start_idx = None
            for idx, (y, text, has_pattern) in enumerate(potential_blocks):
                if has_pattern:
                    caption_start_idx = idx
                    break
            
            if caption_start_idx is None:
                return None
            
            # Collect all blocks starting from the caption pattern
            # Continue collecting until we hit a large gap or non-caption-looking text
            for idx in range(caption_start_idx, len(potential_blocks)):
                y, text, has_pattern = potential_blocks[idx]
                
                if idx == caption_start_idx:
                    # First block - always add
                    caption_blocks.append((y, text))
                else:
                    # Check gap from previous block
                    prev_y = caption_blocks[-1][0]
                    vertical_gap = y - prev_y
                    
                    # Much more lenient gap tolerance - up to 80 pixels
                    # Captions can have significant line breaks
                    if vertical_gap < 80:
                        caption_blocks.append((y, text))
                    elif vertical_gap < 150:
                        # Check if this looks like caption continuation
                        text_stripped = text.strip()
                        if text_stripped:
                            # Continue if starts with lowercase, punctuation, or number
                            first_char = text_stripped[0]
                            if (first_char.islower() or 
                                first_char in ',.;:' or
                                first_char.isdigit() or
                                text_stripped.startswith('(') or
                                text_stripped.startswith('[')):
                                caption_blocks.append((y, text))
                            else:
                                # Might be start of new paragraph - check if previous caption ended with period
                                prev_text = caption_blocks[-1][1].strip()
                                if prev_text.endswith('.') or prev_text.endswith(':'):
                                    # Previous caption might be complete, but check if this could still be part
                                    # Only continue if it looks like continuation (lowercase, etc.)
                                    if first_char.islower():
                                        caption_blocks.append((y, text))
                                    else:
                                        break
                                else:
                                    # Previous caption didn't end properly, might continue
                                    caption_blocks.append((y, text))
                        else:
                            break
                    else:
                        # Gap is too large (>150px), stop collecting
                        break
            
            if not caption_blocks:
                return None
            
            # Sort by vertical position (top to bottom)
            caption_blocks.sort(key=lambda x: x[0])
            
            # Combine all caption blocks into one caption
            full_caption = " ".join(block_text for _, block_text in caption_blocks)
            
            # DEBUG: Log caption extraction for debugging
            if len(caption_blocks) > 1:
                print(f"DEBUG: Caption extracted from {len(caption_blocks)} blocks, total length: {len(full_caption)} chars")
                if len(full_caption) > 200:
                    print(f"DEBUG: Caption preview (first 200 chars): {full_caption[:200]}...")

            # Clean up the caption
            full_caption = full_caption.replace('\n', ' ').replace('\r', ' ')
            # Remove extra whitespace
            full_caption = ' '.join(full_caption.split())
            
            # CRITICAL: Validate caption - filter out binary data
            # Check for binary patterns in caption
            if "b'\\x89" in full_caption or 'b"\\x89' in full_caption:
                print(f"WARNING: Caption contains binary PNG pattern, filtering it out")
                return None
            if "b'\\r\\n\\x1a" in full_caption or 'b"\\r\\n\\x1a' in full_caption:
                print(f"WARNING: Caption contains PNG header continuation, filtering it out")
                return None
            if "b'\\x" in full_caption or 'b"\\x' in full_caption:
                print(f"WARNING: Caption contains byte string patterns, filtering it out")
                return None
            # Check if caption looks like binary data
            if not self._is_valid_text(full_caption):
                print(f"WARNING: Caption failed validation (likely binary data), filtering it out")
                return None
            
            # Increase limit to 1000 chars to accommodate longer captions
            # Many figure captions can be quite long and detailed
            if len(full_caption) > 1000:
                # Try to truncate at a sentence boundary
                truncated = full_caption[:1000]
                last_period = truncated.rfind('.')
                last_colon = truncated.rfind(':')
                last_semicolon = truncated.rfind(';')
                last_punctuation = max(last_period, last_colon, last_semicolon)
                if last_punctuation > 900:  # Only truncate at punctuation if it's not too early
                    full_caption = truncated[:last_punctuation + 1]
                else:
                    # Truncate at word boundary
                    last_space = truncated.rfind(' ')
                    if last_space > 900:
                        full_caption = truncated[:last_space] + "..."
                    else:
                        full_caption = truncated + "..."
            
            return full_caption.strip() if full_caption.strip() else None
            
        except Exception as e:
            print(f"Error in _extract_caption_from_page: {e}")
            return None
    
    async def _save_text_chunks(self, chunks: List[Dict[str, Any]], document_id: int):
        """
        Save text chunks to database with embeddings.
        
        TODO: Implement chunk storage
        - Generate embeddings
        - Store in database
        - Link related images/tables in metadata
        """
        saved_count = 0
        skipped_count = 0
        
        for chunk in chunks:
            try:
                content = chunk.get("content", "")
                
                # Validate chunk content before saving - skip if binary data
                if not content or not content.strip():
                    skipped_count += 1
                    continue
                
                # Aggressive validation - check for ANY binary patterns
                import re
                content_str = str(content)
                should_skip_chunk = False
                
                # Check for Python byte string representations (b'...)
                if "b'\\x" in content_str or 'b"\\x' in content_str:
                    print(f"Skipping chunk {chunk.get('chunk_index', '?')} on page {chunk.get('page_number', '?')} - contains Python byte string (binary data)")
                    should_skip_chunk = True
                
                # Check for corrupted hex patterns
                if not should_skip_chunk:
                    corrupted_patterns = [
                        r'x[0-9a-fA-F]{3,}',  # x9477, x834, etc.
                        r'[a-z0-9]{1,2}[0-9a-fA-F]{1,2}x0x0b',  # ub0x0b, 5b0x0b
                    ]
                    for pattern in corrupted_patterns:
                        if re.search(pattern, content_str, re.IGNORECASE):
                            print(f"Skipping chunk {chunk.get('chunk_index', '?')} on page {chunk.get('page_number', '?')} - contains corrupted hex pattern ({pattern})")
                            should_skip_chunk = True
                            break
                
                # Final validation check
                if not should_skip_chunk and not self._is_valid_text(content_str):
                    print(f"Skipping chunk {chunk.get('chunk_index', '?')} on page {chunk.get('page_number', '?')} - validation failed (length: {len(content_str)}, preview: {repr(content_str[:100])})")
                    should_skip_chunk = True
                
                if should_skip_chunk:
                    skipped_count += 1
                    continue
                
                await self.vector_store.store_chunk(
                    content=content,
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    metadata=chunk.get("metadata", {})
                )
                saved_count += 1
            except Exception as e:
                print(f"Error saving chunk: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"Saved {saved_count} chunks, skipped {skipped_count} chunks (binary data or invalid)")
    
    async def _save_image(
        self, 
        image_data: Any, 
        document_id: int, 
        page_number: int,
        metadata: Dict[str, Any],
        caption: str = None
    ) -> DocumentImage:
        """
        Save an extracted image.
        
        TODO: Implement image saving
        - Save image file to disk
        - Create DocumentImage record
        - Extract caption if available
        """
        try:
            # Generate unique filename
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"
            image_path = os.path.join(settings.UPLOAD_DIR, "images", filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Convert image data to PIL Image if needed
            if isinstance(image_data, Image.Image):
                img = image_data
            elif isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif hasattr(image_data, 'image'):
                # Docling image object
                img = image_data.image
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
            else:
                # Try to convert to PIL Image
                img = Image.open(io.BytesIO(image_data))
            
            # Save image
            img.save(image_path, "PNG")
            
            # Extract caption if not provided and available from image_data
            if not caption:
                # Try caption attribute (property)
                if hasattr(image_data, 'caption'):
                    try:
                        caption_val = image_data.caption
                        # Check if it's a method (callable) or a property
                        if callable(caption_val):
                            caption = str(caption_val())
                        elif caption_val:
                            caption = str(caption_val)
                    except Exception as e:
                        print(f"Error extracting caption attribute: {e}")
                
                # Try title attribute (property)
                if not caption and hasattr(image_data, 'title'):
                    try:
                        title_val = image_data.title
                        # Check if it's a method (callable) or a property
                        if callable(title_val):
                            caption = str(title_val())
                        elif title_val:
                            caption = str(title_val)
                    except Exception as e:
                        print(f"Error extracting title attribute: {e}")
                
                # Try get_caption method
                if not caption and hasattr(image_data, 'get_caption'):
                    try:
                        caption_result = image_data.get_caption()
                        if caption_result:
                            caption = str(caption_result)
                    except Exception as e:
                        print(f"Error calling get_caption: {e}")
                
                # Try get_title method
                if not caption and hasattr(image_data, 'get_title'):
                    try:
                        title_result = image_data.get_title()
                        if title_result:
                            caption = str(title_result)
                    except Exception as e:
                        print(f"Error calling get_title: {e}")
            
            # If still no caption, try to extract from metadata
            if not caption and metadata:
                caption = metadata.get('caption') or metadata.get('title')
                if caption:
                    caption = str(caption)
            
            # CRITICAL: Validate caption before saving - filter out binary data
            if caption:
                caption_str = str(caption).strip()
                # Check for binary patterns
                if "b'\\x89" in caption_str or 'b"\\x89' in caption_str:
                    print(f"WARNING: Image caption contains binary PNG pattern, removing it")
                    caption = None
                elif "b'\\r\\n\\x1a" in caption_str or 'b"\\r\\n\\x1a' in caption_str:
                    print(f"WARNING: Image caption contains PNG header continuation, removing it")
                    caption = None
                elif "b'\\x" in caption_str or 'b"\\x' in caption_str:
                    print(f"WARNING: Image caption contains byte string patterns, removing it")
                    caption = None
                elif not self._is_valid_text(caption_str):
                    print(f"WARNING: Image caption failed validation (likely binary data), removing it")
                    caption = None

            # Create database record
            document_image = DocumentImage(
                document_id=document_id,
                file_path=image_path,
                page_number=page_number,
                caption=caption,
                width=img.width,
                height=img.height,
                extra_metadata=metadata
            )
            
            self.db.add(document_image)
            self.db.commit()
            self.db.refresh(document_image)
            
            return document_image
            
        except Exception as e:
            print(f"Error saving image: {e}")
            self.db.rollback()
            raise
    
    async def _save_table(
        self,
        table_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save an extracted table.
        
        TODO: Implement table saving
        - Render table as image
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        """
        try:
            # Generate unique filename
            table_id = str(uuid.uuid4())
            filename = f"{table_id}.png"
            table_image_path = os.path.join(settings.UPLOAD_DIR, "tables", filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(table_image_path), exist_ok=True)
            
            # Extract structured data from Docling TableItem
            # Try multiple methods to get complete table data, preserving original structure
            structured_data = None
            rows = 0
            columns = 0
            
            # Debug: Check what attributes the table has
            print(f"Table item type: {type(table_data)}")
            print(f"Table attributes: {[attr for attr in dir(table_data) if not attr.startswith('_')][:20]}")
            
            # Method 0: Try to get raw table structure first (most accurate)
            try:
                # Check for raw table structure methods
                if hasattr(table_data, 'get_cells') or hasattr(table_data, 'cells'):
                    cells = None
                    if hasattr(table_data, 'get_cells'):
                        cells = table_data.get_cells()
                    elif hasattr(table_data, 'cells'):
                        cells = table_data.cells
                    
                    if cells:
                        # Reconstruct table from cells
                        # Cells might be a list of cell objects with row/col coordinates
                        if isinstance(cells, list) and len(cells) > 0:
                            # Find max row and col
                            max_row = 0
                            max_col = 0
                            cell_dict = {}
                            
                            for cell in cells:
                                if hasattr(cell, 'row') and hasattr(cell, 'col'):
                                    row_idx = cell.row
                                    col_idx = cell.col
                                    max_row = max(max_row, row_idx)
                                    max_col = max(max_col, col_idx)
                                    # Get cell value
                                    if hasattr(cell, 'value'):
                                        cell_dict[(row_idx, col_idx)] = str(cell.value) if cell.value is not None else ""
                                    elif hasattr(cell, 'text'):
                                        cell_dict[(row_idx, col_idx)] = str(cell.text) if cell.text is not None else ""
                            
                            # Build structured data matrix
                            if max_row >= 0 and max_col >= 0:
                                structured_data = []
                                for r in range(max_row + 1):
                                    row = []
                                    for c in range(max_col + 1):
                                        row.append(cell_dict.get((r, c), ""))
                                    structured_data.append(row)
                                
                                rows = len(structured_data)
                                columns = max(len(row) for row in structured_data) if structured_data else 0
                                print(f"Got table via cells: {rows} rows x {columns} columns")
            except Exception as e:
                print(f"Error extracting from cells: {e}")
            
            # Method 1: Try export_to_dataframe first (most reliable) if pandas is available
            # NOTE: export_to_dataframe() now requires 'doc' argument (deprecation warning)
            if not structured_data:
                try:
                    if hasattr(table_data, 'export_to_dataframe'):
                        try:
                            import pandas as pd
                            # Pass 'doc' argument to fix deprecation warning and potentially get better results
                            try:
                                df = table_data.export_to_dataframe(doc)
                            except TypeError:
                                # Fallback if doc argument not supported in this version
                                df = table_data.export_to_dataframe()
                            
                            if df is not None and not df.empty:
                                rows = len(df)
                                columns = len(df.columns)
                                # Convert DataFrame to list of lists for JSON storage
                                # Include column names as first row, ensure all cells are converted to strings
                                structured_data = [df.columns.tolist()]
                                # Convert all values to strings to preserve data, handling NaN values
                                for idx, row in df.iterrows():
                                    row_data = []
                                    for col in df.columns:
                                        val = row[col]
                                        # Handle NaN, None, and other special values
                                        if pd.isna(val):
                                            row_data.append("")
                                        elif val is None:
                                            row_data.append("")
                                        else:
                                            # Preserve original value as string, including scientific notation
                                            row_data.append(str(val))
                                    structured_data.append(row_data)
                                print(f"Got table via export_to_dataframe: {rows} rows x {columns} columns, total cells: {rows * columns}")
                                # Validate data completeness
                                if structured_data:
                                    total_cells = sum(len(row) for row in structured_data)
                                    expected_cells = (rows + 1) * columns  # +1 for header
                                    print(f"Data completeness: {total_cells}/{expected_cells} cells extracted")
                        except ImportError:
                            print("Pandas not available, trying other methods")
                        except Exception as e:
                            print(f"Error in export_to_dataframe: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"Error checking export_to_dataframe: {e}")
            
            # Method 2: Try export_to_dict() which might be more reliable
            if rows == 0 or columns == 0 or not structured_data:
                try:
                    if hasattr(table_data, 'export_to_dict'):
                        dict_data = table_data.export_to_dict()
                        if dict_data:
                            if isinstance(dict_data, dict):
                                if 'rows' in dict_data:
                                    rows_data = dict_data['rows']
                                    if isinstance(rows_data, list) and len(rows_data) > 0:
                                        structured_data = rows_data
                                        rows = len(rows_data)
                                        if isinstance(rows_data[0], list):
                                            columns = len(rows_data[0])
                                elif 'data' in dict_data:
                                    data = dict_data['data']
                                    if isinstance(data, list):
                                        structured_data = data
                                        rows = len(data)
                                        if rows > 0 and isinstance(data[0], list):
                                            columns = len(data[0])
                            print(f"Got table via export_to_dict: {rows} rows x {columns} columns")
                except Exception as e:
                    print(f"Error in export_to_dict: {e}")
            
            # Method 3: Try accessing .data attribute directly (might have raw structure)
            if rows == 0 or columns == 0 or not structured_data:
                try:
                    if hasattr(table_data, 'data'):
                        data_obj = table_data.data
                        print(f"Table .data type: {type(data_obj)}")
                        
                        # Check what type data is
                        if hasattr(data_obj, 'values') and hasattr(data_obj, 'columns'):
                            # It's a DataFrame-like object
                            try:
                                import pandas as pd
                                if isinstance(data_obj, pd.DataFrame):
                                    # Convert DataFrame properly
                                    rows = len(data_obj)
                                    columns = len(data_obj.columns)
                                    structured_data = [data_obj.columns.tolist()]
                                    for idx, row in data_obj.iterrows():
                                        row_data = []
                                        for col in data_obj.columns:
                                            val = row[col]
                                            if pd.isna(val):
                                                row_data.append("")
                                            elif val is None:
                                                row_data.append("")
                                            else:
                                                row_data.append(str(val))
                                        structured_data.append(row_data)
                                    print(f"Got table via .data (DataFrame): {rows} rows x {columns} columns")
                            except:
                                rows = len(data_obj)
                                columns = len(data_obj.columns) if hasattr(data_obj, 'columns') else 0
                                if hasattr(data_obj, 'values'):
                                    structured_data = data_obj.values.tolist() if hasattr(data_obj.values, 'tolist') else data_obj.values
                        elif isinstance(data_obj, list):
                            structured_data = data_obj
                            rows = len(data_obj)
                            if rows > 0:
                                if isinstance(data_obj[0], list):
                                    columns = max(len(row) for row in data_obj) if data_obj else 0
                                    # Pad rows to same length
                                    for row in structured_data:
                                        while len(row) < columns:
                                            row.append("")
                                elif isinstance(data_obj[0], dict):
                                    columns = len(data_obj[0].keys()) if data_obj[0] else 0
                        elif isinstance(data_obj, dict):
                            structured_data = data_obj
                            if 'rows' in data_obj:
                                rows_data = data_obj['rows']
                                rows = len(rows_data) if isinstance(rows_data, list) else 0
                                if rows > 0 and isinstance(rows_data[0], list):
                                    columns = max(len(row) for row in rows_data) if rows_data else 0
                                    # Pad rows
                                    for row in rows_data:
                                        while len(row) < columns:
                                            row.append("")
                        print(f"Got table via .data: {rows} rows x {columns} columns")
                except Exception as e:
                    print(f"Error accessing .data: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Method 4: Try export_to_markdown and parse (last resort, but often most complete)
            if rows == 0 or columns == 0 or not structured_data:
                try:
                    if hasattr(table_data, 'export_to_markdown'):
                        markdown = table_data.export_to_markdown()
                        if markdown:
                            print(f"Markdown table length: {len(markdown)} chars")
                            # Parse markdown table to get data - be more thorough
                            lines = [line for line in markdown.split('\n') if line.strip()]
                            parsed_rows = []
                            
                            for line in lines:
                                line = line.strip()
                                # Skip separator lines (but count them to understand structure)
                                if line.startswith('|---') or line.startswith('|:---') or line.startswith('|---:') or line.startswith('| ---'):
                                    continue
                                
                                # Parse cells more carefully - preserve empty cells
                                if '|' in line:
                                    # Split by | - be careful with empty cells
                                    # Markdown tables use | to separate cells, even empty ones
                                    parts = line.split('|')
                                    cells = []
                                    
                                    # Process each part
                                    for i, part in enumerate(parts):
                                        part = part.strip()
                                        # First and last parts might be empty (markdown format)
                                        if i == 0 and part == '':
                                            continue  # Skip leading empty
                                        if i == len(parts) - 1 and part == '':
                                            continue  # Skip trailing empty
                                        cells.append(part)
                                    
                                    if cells:
                                        parsed_rows.append(cells)
                            
                            if parsed_rows:
                                structured_data = parsed_rows
                                rows = len(parsed_rows)
                                if rows > 0:
                                    # Find max columns across all rows
                                    columns = max(len(row) for row in parsed_rows)
                                    # Pad rows that are shorter to ensure consistent structure
                                    for row in parsed_rows:
                                        while len(row) < columns:
                                            row.append("")
                                
                                # Log detailed info
                                non_empty_cells = sum(1 for row in parsed_rows for cell in row if cell and cell.strip())
                                total_cells = rows * columns
                                print(f"Got table via export_to_markdown: {rows} rows x {columns} columns")
                                print(f"Markdown extraction: {non_empty_cells}/{total_cells} non-empty cells")
                            else:
                                # Fallback: store markdown as-is for manual parsing later
                                structured_data = {"markdown": markdown, "raw": True}
                                # Try to estimate dimensions from markdown
                                table_lines = [l for l in lines if '|' in l and not (l.startswith('|---') or l.startswith('|:---'))]
                                if table_lines:
                                    rows = len(table_lines)
                                    # Count columns from first data row
                                    first_line = table_lines[0] if table_lines else ""
                                    columns = max(first_line.count('|') - 1, 1)
                                print(f"Stored markdown table as raw: {rows} rows x {columns} columns (estimated)")
                except Exception as e:
                    print(f"Error in export_to_markdown: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Validate extracted data and update dimensions
            if structured_data:
                if isinstance(structured_data, list):
                    actual_rows = len(structured_data)
                    if actual_rows > 0:
                        if isinstance(structured_data[0], list):
                            actual_cols = max(len(row) for row in structured_data)
                            # Ensure all rows have the same number of columns
                            for row in structured_data:
                                while len(row) < actual_cols:
                                    row.append("")
                            
                            # Update dimensions to match actual data
                            rows = actual_rows
                            columns = actual_cols
                            
                            print(f"Final extracted table: {rows} rows x {columns} columns")
                            
                            # Check for empty rows or missing data
                            empty_cells = sum(1 for row in structured_data for cell in row if not cell or (isinstance(cell, str) and cell.strip() == ""))
                            total_cells = sum(len(row) for row in structured_data)
                            non_empty_cells = total_cells - empty_cells
                            
                            print(f"Data quality: {non_empty_cells}/{total_cells} non-empty cells ({100*non_empty_cells/total_cells:.1f}%)")
                            
                            if empty_cells > total_cells * 0.5:  # More than 50% empty
                                print(f"Warning: {empty_cells}/{total_cells} cells are empty - table extraction may be incomplete!")
                            
                            # Log sample of first few rows for debugging
                            if rows > 0:
                                print(f"Sample row 0: {structured_data[0][:min(5, columns)]}")
                                if rows > 1:
                                    print(f"Sample row 1: {structured_data[1][:min(5, columns)]}")
                        else:
                            print(f"Final extracted table: {actual_rows} rows, data structure: {type(structured_data[0])}")
                            rows = actual_rows
                            columns = 1
                    else:
                        print(f"Warning: Table has no rows!")
                        rows = 0
                        columns = 0
                elif isinstance(structured_data, dict):
                    # Handle dict format (e.g., from export_to_dict or markdown fallback)
                    if 'markdown' in structured_data:
                        print(f"Table stored as markdown (raw format)")
                        # Keep dimensions as estimated
                    else:
                        print(f"Table stored as dict: {structured_data.keys()}")
                else:
                    print(f"Final extracted table: data type is {type(structured_data)}, not a list")
                    rows = 0
                    columns = 0
            else:
                print(f"Warning: No structured data extracted for table!")
                rows = 0
                columns = 0
            
            # Render table as image - prefer Docling's native rendering
            # Try get_image() first (most accurate, shows table as it appears in PDF)
            table_image_rendered = False
            try:
                if hasattr(table_data, 'get_image'):
                    try:
                        # get_image() might require doc argument
                        table_img = table_data.get_image(doc)
                        if table_img:
                            if isinstance(table_img, Image.Image):
                                table_img.save(table_image_path, "PNG")
                                table_image_rendered = True
                                print(f"Rendered table using get_image(doc)")
                            elif isinstance(table_img, bytes):
                                img = Image.open(io.BytesIO(table_img))
                                img.save(table_image_path, "PNG")
                                table_image_rendered = True
                                print(f"Rendered table using get_image(doc) bytes")
                    except TypeError:
                        # Try without doc argument
                        try:
                            table_img = table_data.get_image()
                            if table_img:
                                if isinstance(table_img, Image.Image):
                                    table_img.save(table_image_path, "PNG")
                                    table_image_rendered = True
                                    print(f"Rendered table using get_image()")
                        except:
                            pass
                    except Exception as e:
                        print(f"Error in get_image(): {e}")
            except Exception as e:
                print(f"Error checking get_image(): {e}")
            
            # Fallback: Use provided image attribute
            if not table_image_rendered and hasattr(table_data, 'image') and table_data.image:
                # Use provided image
                img = table_data.image
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                elif not isinstance(img, Image.Image):
                    img = Image.open(io.BytesIO(img))
                img.save(table_image_path, "PNG")
                table_image_rendered = True
                print(f"Rendered table using .image attribute")
            
            # Last resort: Render from structured data
            if not table_image_rendered and structured_data and isinstance(structured_data, list) and len(structured_data) > 0:
                # Render table from structured data - use actual data dimensions
                try:
                    from PIL import ImageDraw, ImageFont
                    
                    # Get actual dimensions from data
                    actual_rows = len(structured_data)
                    if isinstance(structured_data[0], list):
                        actual_cols = max(len(row) for row in structured_data) if structured_data else 0
                    else:
                        actual_cols = 1
                    
                    # Update rows/columns to match actual data
                    rows = actual_rows
                    columns = actual_cols
                    
                    # Calculate cell width dynamically based on content
                    # First, find the maximum text length in each column
                    column_max_lengths = [0] * columns
                    for row in structured_data:
                        if isinstance(row, list):
                            for col_idx, cell in enumerate(row[:columns]):
                                cell_text = str(cell) if cell else ""
                                column_max_lengths[col_idx] = max(column_max_lengths[col_idx], len(cell_text))
                    
                    # Calculate cell widths based on content (minimum width, but expand for long text)
                    font_size = 11
                    base_char_width = 7  # Approximate character width in pixels
                    min_cell_width = 120
                    cell_widths = []
                    
                    for col_idx, max_len in enumerate(column_max_lengths):
                        # Calculate width needed for this column's longest text
                        # Add padding (20px) and ensure minimum width
                        calculated_width = max(min_cell_width, (max_len * base_char_width) + 40)
                        # Cap at reasonable maximum to avoid extremely wide cells
                        calculated_width = min(calculated_width, 500)
                        cell_widths.append(calculated_width)
                    
                    # Calculate total image width
                    img_width = sum(cell_widths) + 20
                    
                    # Calculate cell height - allow for multi-line text
                    cell_height = 35  # Base height
                    # Check if any cells need more height (for wrapping)
                    max_lines_per_cell = 1
                    for row in structured_data:
                        if isinstance(row, list):
                            for col_idx, cell in enumerate(row[:columns]):
                                cell_text = str(cell) if cell else ""
                                # Estimate lines needed (rough calculation)
                                if col_idx < len(cell_widths):
                                    chars_per_line = (cell_widths[col_idx] - 20) // base_char_width
                                    if chars_per_line > 0:
                                        lines_needed = (len(cell_text) + chars_per_line - 1) // chars_per_line
                                        max_lines_per_cell = max(max_lines_per_cell, lines_needed)
                    
                    # Adjust cell height for multi-line text
                    cell_height = max(cell_height, 25 + (max_lines_per_cell - 1) * 15)
                    img_height = rows * cell_height + 20
                    
                    # Limit maximum image size (for very large tables)
                    max_width = 5000  # Increased from 3000
                    max_height = 6000  # Increased from 4000
                    if img_width > max_width:
                        # Scale down proportionally
                        scale_factor = max_width / img_width
                        cell_widths = [int(w * scale_factor) for w in cell_widths]
                        img_width = max_width
                    if img_height > max_height:
                        cell_height = max_height // rows
                        img_height = max_height
                    
                    img = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to use default font
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                    except:
                        try:
                            font = ImageFont.load_default()
                        except:
                            font = None
                    
                    # Draw table - render ALL rows and columns
                    y = 10
                    for row_idx, row in enumerate(structured_data):
                        if y + cell_height > img_height:
                            print(f"Warning: Table too large, stopping at row {row_idx}/{rows}")
                            break
                        
                        x = 10
                        # Ensure row has enough columns
                        if isinstance(row, list):
                            row_data = row
                        else:
                            row_data = [str(row)]
                        
                        # Pad row if needed
                        while len(row_data) < columns:
                            row_data.append("")
                        
                        for col_idx in range(columns):
                            if col_idx >= len(cell_widths):
                                break
                            
                            current_cell_width = cell_widths[col_idx]
                            if x + current_cell_width > img_width:
                                break
                            
                            cell = row_data[col_idx] if col_idx < len(row_data) else ""
                            cell_text = str(cell) if cell else ""
                            
                            # Draw cell border
                            draw.rectangle([x, y, x + current_cell_width, y + cell_height], outline='black')
                            
                            # Draw cell text - wrap text if needed instead of truncating
                            text_x = x + 5
                            text_y = y + 8
                            
                            # Calculate how many characters fit per line
                            chars_per_line = (current_cell_width - 10) // base_char_width
                            
                            if chars_per_line > 0 and len(cell_text) > chars_per_line:
                                # Wrap text into multiple lines
                                words = cell_text.split(' ')
                                lines = []
                                current_line = ""
                                
                                for word in words:
                                    test_line = current_line + (" " if current_line else "") + word
                                    if len(test_line) <= chars_per_line:
                                        current_line = test_line
                                    else:
                                        if current_line:
                                            lines.append(current_line)
                                        # If word itself is longer than line, break it
                                        if len(word) > chars_per_line:
                                            # Break long word
                                            while len(word) > chars_per_line:
                                                lines.append(word[:chars_per_line])
                                                word = word[chars_per_line:]
                                            current_line = word
                                        else:
                                            current_line = word
                                
                                if current_line:
                                    lines.append(current_line)
                                
                                # Draw each line
                                line_height = 14
                                for line_idx, line in enumerate(lines[:5]):  # Max 5 lines per cell
                                    if text_y + (line_idx * line_height) + line_height > y + cell_height:
                                        break
                                    if font:
                                        draw.text((text_x, text_y + (line_idx * line_height)), line, fill='black', font=font)
                                    else:
                                        draw.text((text_x, text_y + (line_idx * line_height)), line, fill='black')
                            else:
                                # Single line text - no truncation, show full text
                                if font:
                                    draw.text((text_x, text_y), cell_text, fill='black', font=font)
                                else:
                                    draw.text((text_x, text_y), cell_text, fill='black')
                            
                            x += current_cell_width
                        y += cell_height
                    
                    img.save(table_image_path, "PNG")
                    print(f"Rendered table image: {rows} rows x {columns} columns, size: {img_width}x{img_height}")
                except Exception as e:
                    print(f"Error rendering table image: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create a placeholder image
                    img = Image.new('RGB', (400, 200), color='lightgray')
                    img.save(table_image_path, "PNG")
            else:
                # Create placeholder image
                img = Image.new('RGB', (400, 200), color='lightgray')
                img.save(table_image_path, "PNG")
            
            # Extract caption
            caption = None
            if hasattr(table_data, 'caption') and table_data.caption:
                caption = str(table_data.caption)
            elif hasattr(table_data, 'title') and table_data.title:
                caption = str(table_data.title)
            
            # Create database record
            document_table = DocumentTable(
                document_id=document_id,
                image_path=table_image_path,
                data=structured_data,
                page_number=page_number,
                caption=caption,
                rows=rows,
                columns=columns,
                extra_metadata=metadata
            )
            
            self.db.add(document_table)
            self.db.commit()
            self.db.refresh(document_table)
            
            return document_table
            
        except Exception as e:
            print(f"Error saving table: {e}")
            self.db.rollback()
            raise
    
    async def _update_document_status(
        self, 
        document_id: int, 
        status: str, 
        error_message: str = None
    ):
        """
        Update document processing status.
        
        This is implemented as an example.
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.processing_status = status
            if error_message:
                document.error_message = error_message
            self.db.commit()
