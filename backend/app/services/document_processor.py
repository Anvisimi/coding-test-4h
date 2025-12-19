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
            
            # If we have full text, chunk it (even if we have pages, we can use full text for better context)
            if full_text and full_text.strip():
                # Split full text by pages if possible, otherwise chunk as whole
                chunks = self._chunk_text(full_text, document_id, 1)
                if chunks:
                    await self._save_text_chunks(chunks, document_id)
                    text_chunks_count += len(chunks)
                    print(f"Created {len(chunks)} chunks from full text")
            
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
            
            # Process each page
            for page_idx, page in enumerate(pages, start=1):
                # Extract text and chunk it - try different ways to get text
                page_text = None
                page_num = page_idx
                
                # Get page number from page object if available
                if hasattr(page, 'page'):
                    page_num = page.page
                elif hasattr(page, 'page_number'):
                    page_num = page.page_number
                
                # Try to get text from page
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
                            page_text_parts.append(str(item.text))
                    if page_text_parts:
                        page_text = '\n\n'.join(page_text_parts)
                
                if page_text and page_text.strip():
                    chunks = self._chunk_text(str(page_text), document_id, page_num)
                    if chunks:
                        await self._save_text_chunks(chunks, document_id)
                        text_chunks_count += len(chunks)
                
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
                        if hasattr(image_item, 'caption') and image_item.caption:
                            extracted_caption = str(image_item.caption)
                        elif hasattr(image_item, 'title') and image_item.title:
                            extracted_caption = str(image_item.title)
                        elif hasattr(image_item, 'get_caption'):
                            try:
                                extracted_caption = str(image_item.get_caption())
                            except:
                                pass
                        
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
                                    # Get text blocks near the image (above it, within 50 pixels)
                                    text_blocks = page.get_text("blocks")
                                    for block in text_blocks:
                                        block_rect = fitz.Rect(block[:4])
                                        block_text = block[4] if len(block) > 4 else ""
                                        # Check if text block is above the image (within 50 pixels)
                                        if (block_rect.y1 < img_bbox.y0 + 50 and 
                                            abs(block_rect.x0 - img_bbox.x0) < 100 and
                                            block_text):
                                            # Look for "Figure" or "Fig" patterns
                                            if "Figure" in block_text or "Fig." in block_text:
                                                # Extract caption (first line or first 100 chars)
                                                caption = block_text.split('\n')[0][:100].strip()
                                                break
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
                                                text_blocks = page.get_text("blocks")
                                                for block in text_blocks:
                                                    block_rect = fitz.Rect(block[:4])
                                                    block_text = block[4] if len(block) > 4 else ""
                                                    if (block_rect.y1 < bbox.y0 + 50 and 
                                                        abs(block_rect.x0 - bbox.x0) < 100 and
                                                        block_text and ("Figure" in block_text or "Fig." in block_text)):
                                                        caption = block_text.split('\n')[0][:100].strip()
                                                        break
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
                                                # Get text blocks near the figure region
                                                text_blocks = page.get_text("blocks")
                                                for block in text_blocks:
                                                    block_rect = fitz.Rect(block[:4])
                                                    # Check if text block is near the figure (within 50 pixels)
                                                    if (block_rect.y1 < clip_rect.y0 + 50 and 
                                                        abs(block_rect.x0 - clip_rect.x0) < 100):
                                                        block_text = block[4] if len(block) > 4 else ""
                                                        # Look for "Figure" or "Fig" patterns
                                                        if block_text and ("Figure" in block_text or "Fig." in block_text):
                                                            # Extract caption (first line or first 100 chars)
                                                            caption = block_text.split('\n')[0][:100].strip()
                                                            break
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
        
        chunks = []
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
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
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "metadata": {}
                    })
                    chunk_index += 1
                
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
                            if current_chunk:
                                chunks.append({
                                    "content": current_chunk + ".",
                                    "page_number": page_number,
                                    "chunk_index": chunk_index,
                                    "metadata": {}
                                })
                                chunk_index += 1
                            
                            # Start new chunk with overlap
                            if chunk_overlap > 0 and chunks:
                                # Get last few words from previous chunk for overlap
                                prev_words = chunks[-1]["content"].split()[-chunk_overlap//10:]
                                current_chunk = " ".join(prev_words) + " " + sentence
                            else:
                                current_chunk = sentence
                    
                    # Add remaining chunk
                    if current_chunk:
                        current_chunk += "." if not current_chunk.endswith(".") else ""
                else:
                    # Start new chunk with overlap
                    if chunk_overlap > 0 and chunks:
                        prev_words = chunks[-1]["content"].split()[-chunk_overlap//10:]
                        current_chunk = " ".join(prev_words) + "\n\n" + para
                    else:
                        current_chunk = para
        
        # Add final chunk
        if current_chunk:
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
        
        TODO: Implement chunk storage
        - Generate embeddings
        - Store in database
        - Link related images/tables in metadata
        """
        for chunk in chunks:
            try:
                await self.vector_store.store_chunk(
                    content=chunk["content"],
                    document_id=document_id,
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    metadata=chunk.get("metadata", {})
                )
            except Exception as e:
                print(f"Error saving chunk: {e}")
                continue
    
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
                if hasattr(image_data, 'caption') and image_data.caption:
                    caption = str(image_data.caption)
                elif hasattr(image_data, 'title') and image_data.title:
                    caption = str(image_data.title)
                elif hasattr(image_data, 'get_caption'):
                    try:
                        caption = str(image_data.get_caption())
                    except:
                        pass
            
            # If still no caption, try to extract from metadata
            if not caption and metadata:
                caption = metadata.get('caption') or metadata.get('title')
            
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
