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
                                {"index": img_idx, "source": "docling"}
                            )
                            images_count += 1
                            successfully_extracted_from_docling = True
                            print(f"Saved image {img_idx} from page {page_num_for_img}")
                        else:
                            print(f"Could not extract image data from picture item {img_idx}")
                    except Exception as e:
                        print(f"Error saving image on page {page_num}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Fallback: Extract images directly from PDF using PyMuPDF if Docling extraction failed
            if not successfully_extracted_from_docling:
                try:
                    print("Using PyMuPDF fallback for image extraction...")
                    pdf_doc = fitz.open(file_path)
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        image_list = page.get_images()
                        for img_idx, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = pdf_doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                # Save image directly using PyMuPDF extracted data
                                await self._save_image(
                                    image_bytes,
                                    document_id,
                                    page_num + 1,
                                    {"index": img_idx, "source": "pymupdf", "xref": xref}
                                )
                                images_count += 1
                                print(f"Saved image {img_idx} from page {page_num + 1} via PyMuPDF")
                            except Exception as e:
                                print(f"Error extracting image from PDF page {page_num + 1}: {e}")
                                continue
                    pdf_doc.close()
                    print(f"Total images extracted via PyMuPDF: {images_count}")
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
        metadata: Dict[str, Any]
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
            
            # Extract caption if available
            caption = None
            if hasattr(image_data, 'caption') and image_data.caption:
                caption = str(image_data.caption)
            elif hasattr(image_data, 'title') and image_data.title:
                caption = str(image_data.title)
            
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
            # Docling TableItem has export_to_dataframe() method
            structured_data = None
            rows = 0
            columns = 0
            
            # Try export_to_dataframe first (most reliable) if pandas is available
            try:
                if hasattr(table_data, 'export_to_dataframe'):
                    try:
                        import pandas as pd
                        df = table_data.export_to_dataframe()
                        if df is not None and not df.empty:
                            rows = len(df)
                            columns = len(df.columns)
                            # Convert DataFrame to list of lists for JSON storage
                            structured_data = df.values.tolist()
                            print(f"Got table via export_to_dataframe: {rows} rows x {columns} columns")
                    except ImportError:
                        print("Pandas not available, trying other methods")
            except Exception as e:
                print(f"Error in export_to_dataframe: {e}")
            
            # Fallback: Try accessing .data attribute directly
            if rows == 0 or columns == 0:
                try:
                    if hasattr(table_data, 'data'):
                        data_obj = table_data.data
                        # Check what type data is
                        if hasattr(data_obj, 'values') and hasattr(data_obj, 'columns'):
                            # It's a DataFrame-like object
                            rows = len(data_obj)
                            columns = len(data_obj.columns) if hasattr(data_obj, 'columns') else 0
                            if hasattr(data_obj, 'values'):
                                structured_data = data_obj.values.tolist() if hasattr(data_obj.values, 'tolist') else data_obj.values
                        elif isinstance(data_obj, list):
                            structured_data = data_obj
                            rows = len(data_obj)
                            if rows > 0:
                                if isinstance(data_obj[0], list):
                                    columns = len(data_obj[0])
                                elif isinstance(data_obj[0], dict):
                                    columns = len(data_obj[0].keys()) if data_obj[0] else 0
                        elif isinstance(data_obj, dict):
                            structured_data = data_obj
                            if 'rows' in data_obj:
                                rows_data = data_obj['rows']
                                rows = len(rows_data) if isinstance(rows_data, list) else 0
                                if rows > 0 and isinstance(rows_data[0], list):
                                    columns = len(rows_data[0])
                        print(f"Got table via .data: {rows} rows x {columns} columns")
                except Exception as e:
                    print(f"Error accessing .data: {e}")
            
            # Last resort: Try export_to_markdown and parse
            if rows == 0 or columns == 0:
                try:
                    if hasattr(table_data, 'export_to_markdown'):
                        markdown = table_data.export_to_markdown()
                        if markdown:
                            # Parse markdown table to get dimensions
                            lines = [line for line in markdown.split('\n') if line.strip() and '|' in line]
                            if len(lines) > 1:  # Header + separator + data rows
                                rows = len(lines) - 2  # Exclude header and separator
                                if rows > 0:
                                    columns = lines[0].count('|') - 1  # Count pipes minus edges
                                structured_data = {"markdown": markdown}
                                print(f"Got table via export_to_markdown: {rows} rows x {columns} columns")
                except Exception as e:
                    print(f"Error in export_to_markdown: {e}")
            
            print(f"Final extracted table: {rows} rows x {columns} columns")
            
            # Render table as image
            if hasattr(table_data, 'image') and table_data.image:
                # Use provided image
                img = table_data.image
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                elif not isinstance(img, Image.Image):
                    img = Image.open(io.BytesIO(img))
                img.save(table_image_path, "PNG")
            elif structured_data:
                # Render table from structured data
                try:
                    from PIL import ImageDraw, ImageFont
                    # Calculate image size based on table dimensions
                    cell_width = 150
                    cell_height = 40
                    img_width = columns * cell_width + 20
                    img_height = rows * cell_height + 20
                    
                    img = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to use default font
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw table
                    y = 10
                    for row_idx, row in enumerate(structured_data[:rows]):
                        x = 10
                        for col_idx, cell in enumerate(row[:columns]):
                            cell_text = str(cell)[:20]  # Truncate long text
                            # Draw cell border
                            draw.rectangle([x, y, x + cell_width, y + cell_height], outline='black')
                            # Draw cell text
                            draw.text((x + 5, y + 10), cell_text, fill='black', font=font)
                            x += cell_width
                        y += cell_height
                    
                    img.save(table_image_path, "PNG")
                except Exception as e:
                    print(f"Error rendering table image: {e}")
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
