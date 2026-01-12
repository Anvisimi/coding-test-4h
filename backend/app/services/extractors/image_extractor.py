"""
Image extraction service from PDF documents.

DONE: Implemented image extraction from Docling
DONE: Implemented image extraction from PyMuPDF (embedded, bbox, vector graphics)
DONE: Implemented caption extraction from PDF text
DONE: Implemented image saving to filesystem and database
"""
from typing import Dict, Any, List, Optional, Set
from sqlalchemy.orm import Session
from app.models.document import DocumentImage
from app.core.config import settings
from app.utils.binary_validator import BinaryValidator
from PIL import Image
import io
import os
import uuid
import fitz  # PyMuPDF


class ImageExtractor:
    """
    Extract images from PDF documents using Docling and PyMuPDF.
    
    Uses multiple extraction strategies:
    1. Docling: Extracts images identified by Docling parser
    2. PyMuPDF embedded: Extracts embedded raster images
    3. PyMuPDF bbox: Renders image regions from bounding boxes
    4. PyMuPDF vector: Extracts vector graphics and figure regions
    
    Also extracts captions from PDF text near images.
    """
    
    def __init__(self, db: Session, binary_validator: BinaryValidator):
        self.db = db
        self.binary_validator = binary_validator
    
    async def extract_and_save_images(
        self,
        doc: Any,
        file_path: str,
        pages: List[Any],
        document_id: int,
        pages_with_tables: Set[int]
    ) -> int:
        """
        Extract images from document using multiple strategies.
        
        Strategy:
        1. Extract from Docling (document-level and page-level pictures)
        2. Extract from PyMuPDF (embedded images, bbox rendering, vector graphics)
        3. Combine results, avoiding duplicates
        
        Args:
            doc: Docling document object
            file_path: Path to PDF file
            pages: List of page objects from Docling
            document_id: Document ID for database storage
            pages_with_tables: Set of page numbers that contain tables (to avoid rendering as images)
            
        Returns:
            Number of images extracted
        """
        images_count = 0
        
        # Step 1: Extract images from Docling
        docling_count = await self._extract_from_docling(doc, pages, document_id)
        images_count += docling_count
        
        # Step 2: Extract images from PyMuPDF (supplements Docling)
        pymupdf_count = await self._extract_from_pymupdf(
            file_path,
            document_id,
            pages_with_tables
        )
        
        # Use maximum count to avoid double-counting
        # PyMuPDF is more comprehensive, so prefer its count if it's higher
        if pymupdf_count > docling_count:
            images_count = pymupdf_count
        else:
            images_count = max(images_count, pymupdf_count)
        
        print(f"Total images extracted: {images_count} (Docling: {docling_count}, PyMuPDF: {pymupdf_count})")
        return images_count
    
    async def _extract_from_docling(
        self,
        doc: Any,
        pages: List[Any],
        document_id: int
    ) -> int:
        """
        Extract images from Docling document and pages.
        
        Docling stores images as "pictures" at both document and page levels.
        This method tries multiple strategies to extract image data from Docling
        PictureItem objects.
        
        Returns:
            Number of images extracted
        """
        images_count = 0
        
        # Get document-level pictures
        doc_pictures = []
        try:
            if hasattr(doc, 'pictures') and doc.pictures:
                doc_pictures = list(doc.pictures) if not isinstance(doc.pictures, list) else doc.pictures
                print(f"Found {len(doc_pictures)} pictures via doc.pictures")
            elif hasattr(doc, 'items') and doc.items:
                doc_pictures = [
                    item for item in doc.items
                    if hasattr(item, 'type') and 'picture' in str(getattr(item, 'type', '')).lower()
                ]
                print(f"Found {len(doc_pictures)} pictures via doc.items")
        except Exception as e:
            print(f"Error getting pictures from document: {e}")
        
        # Process each page for page-level images
        for page_idx, page in enumerate(pages, start=1):
            page_num = self._get_page_number(page, page_idx)
            
            # Get page-level images
            page_images = []
            if hasattr(page, 'images') and page.images:
                page_images = page.images if isinstance(page.images, list) else [page.images]
            elif hasattr(page, 'figures') and page.figures:
                page_images = page.figures if isinstance(page.figures, list) else [page.figures]
            elif hasattr(page, 'pictures') and page.pictures:
                page_images = page.pictures if isinstance(page.pictures, list) else [page.pictures]
            elif hasattr(page, 'items') and page.items:
                page_images = [
                    item for item in page.items
                    if hasattr(item, 'type') and 'picture' in str(getattr(item, 'type', '')).lower()
                ]
            
            # Combine page images with document-level pictures on this page
            all_page_images = page_images.copy()
            for pic in doc_pictures:
                pic_page = getattr(pic, 'page', None) or getattr(pic, 'page_number', None)
                if pic_page == page_num or (pic_page is None and page_idx == 1):
                    all_page_images.append(pic)
            
            # Extract each image
            for img_idx, image_item in enumerate(all_page_images):
                try:
                    img_data, page_num_for_img, caption = self._extract_image_from_docling_item(
                        image_item,
                        doc,
                        page_num
                    )
                    
                    if img_data:
                        await self._save_image(
                            img_data,
                            document_id,
                            page_num_for_img,
                            {"index": img_idx, "source": "docling"},
                            caption=caption
                        )
                        images_count += 1
                        print(f"Saved image {img_idx} from page {page_num_for_img} via Docling")
                except Exception as e:
                    print(f"Error extracting image from Docling on page {page_num}: {e}")
                    continue
        
        return images_count
    
    def _extract_image_from_docling_item(
        self,
        image_item: Any,
        doc: Any,
        default_page_num: int
    ) -> tuple[Optional[Any], int, Optional[str]]:
        """
        Extract image data from Docling PictureItem.
        
        Tries multiple methods to get image data:
        1. get_image(doc) - requires document argument
        2. export_to_image()
        3. Direct attributes (.image, .data, .content, .bytes)
        4. render() method
        
        Also extracts caption from image item.
        
        Returns:
            Tuple of (image_data, page_number, caption)
        """
        img_data = None
        page_num_for_img = default_page_num
        
        # Get page number from image item
        if hasattr(image_item, 'page'):
            page_num_for_img = image_item.page
        elif hasattr(image_item, 'page_number'):
            page_num_for_img = image_item.page_number
        elif hasattr(image_item, 'get_page'):
            page_num_for_img = image_item.get_page()
        
        # Extract caption from Docling picture item
        caption = self._extract_caption_from_docling_item(image_item)
        
        # Strategy 1: get_image(doc) - most reliable for Docling
        if hasattr(image_item, 'get_image'):
            try:
                img_result = image_item.get_image(doc)
                if img_result:
                    if isinstance(img_result, Image.Image):
                        img_data = img_result
                    elif isinstance(img_result, bytes):
                        img_data = img_result
                    elif hasattr(img_result, 'image'):
                        img_data = img_result.image
                    elif hasattr(img_result, 'data'):
                        img_data = img_result.data
                    else:
                        # Try to convert to PIL Image
                        try:
                            img_data = Image.open(io.BytesIO(img_result))
                        except:
                            pass
            except Exception as e:
                print(f"Error in get_image(doc): {e}")
        
        # Strategy 2: export_to_image()
        if not img_data and hasattr(image_item, 'export_to_image'):
            try:
                img_data = image_item.export_to_image()
            except Exception as e:
                print(f"Error in export_to_image: {e}")
        
        # Strategy 3: Direct attributes
        if not img_data and hasattr(image_item, 'image'):
            img_data = image_item.image
        elif not img_data and hasattr(image_item, 'data'):
            img_data = image_item.data
        elif not img_data and hasattr(image_item, 'content'):
            img_data = image_item.content
        elif not img_data and hasattr(image_item, 'bytes'):
            img_data = image_item.bytes
        
        # Strategy 4: render() method
        if not img_data and hasattr(image_item, 'render'):
            try:
                img_data = image_item.render()
            except Exception as e:
                print(f"Error in render(): {e}")
        
        # Strategy 5: Already PIL Image or bytes
        if isinstance(image_item, Image.Image):
            img_data = image_item
        elif isinstance(image_item, bytes):
            img_data = image_item
        
        return img_data, page_num_for_img, caption
    
    def _extract_caption_from_docling_item(self, image_item: Any) -> Optional[str]:
        """
        Extract caption from Docling PictureItem.
        
        Tries multiple methods:
        1. caption attribute (property or method)
        2. title attribute (property or method)
        3. get_caption() method
        4. get_title() method
        
        Returns:
            Caption string or None
        """
        caption = None
        
        # Try caption attribute
        if hasattr(image_item, 'caption'):
            try:
                caption_val = image_item.caption
                if callable(caption_val):
                    caption = str(caption_val())
                elif caption_val:
                    caption = str(caption_val)
            except Exception as e:
                print(f"Error extracting Docling caption attribute: {e}")
        
        # Try title attribute
        if not caption and hasattr(image_item, 'title'):
            try:
                title_val = image_item.title
                if callable(title_val):
                    caption = str(title_val())
                elif title_val:
                    caption = str(title_val)
            except Exception as e:
                print(f"Error extracting Docling title attribute: {e}")
        
        # Try get_caption method
        if not caption and hasattr(image_item, 'get_caption'):
            try:
                caption_result = image_item.get_caption()
                if caption_result:
                    caption = str(caption_result)
            except Exception as e:
                print(f"Error calling Docling get_caption: {e}")
        
        # Try get_title method
        if not caption and hasattr(image_item, 'get_title'):
            try:
                title_result = image_item.get_title()
                if title_result:
                    caption = str(title_result)
            except Exception as e:
                print(f"Error calling Docling get_title: {e}")
        
        # Validate caption
        if caption and not self.binary_validator.validate_caption(caption):
            print(f"WARNING: Docling caption failed validation, removing it")
            caption = None
        
        return caption
    
    async def _extract_from_pymupdf(
        self,
        file_path: str,
        document_id: int,
        pages_with_tables: Set[int]
    ) -> int:
        """
        Extract images from PDF using PyMuPDF.
        
        Uses three methods:
        1. Embedded raster images (extract_image)
        2. Bounding box rendering (get_image_bbox + render)
        3. Vector graphics extraction (get_drawings + render figure regions)
        
        Skips pages that contain tables to avoid rendering tables as images.
        
        Returns:
            Number of images extracted
        """
        images_count = 0
        
        try:
            print("Extracting images via PyMuPDF (embedded + vector graphics)...")
            pdf_doc = fitz.open(file_path)
            seen_image_xrefs: Set[int] = set()  # Track extracted images to avoid duplicates
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_number = page_num + 1
                is_table_page = page_number in pages_with_tables
                
                # Method 1: Extract embedded raster images
                embedded_count = await self._extract_embedded_images(
                    pdf_doc,
                    page,
                    page_number,
                    document_id,
                    seen_image_xrefs
                )
                images_count += embedded_count
                
                # Method 2: Extract images by rendering bounding boxes
                # Skip if page is text-heavy (indicates full page with text, not a figure)
                bbox_count = await self._extract_bbox_images(
                    page,
                    page_number,
                    document_id,
                    seen_image_xrefs,
                    is_table_page
                )
                images_count += bbox_count
                
                # Method 3: Extract vector graphics and figure regions
                # Skip if page contains tables
                vector_count = await self._extract_vector_graphics(
                    page,
                    page_number,
                    document_id,
                    is_table_page
                )
                images_count += vector_count
            
            pdf_doc.close()
        except Exception as e:
            print(f"Error in PyMuPDF extraction: {e}")
            import traceback
            traceback.print_exc()
        
        return images_count
    
    async def _extract_embedded_images(
        self,
        pdf_doc: Any,
        page: Any,
        page_number: int,
        document_id: int,
        seen_image_xrefs: Set[int]
    ) -> int:
        """
        Extract embedded raster images from PDF page.
        
        Uses PyMuPDF's extract_image() method to get embedded images.
        Tracks xrefs to avoid extracting the same image multiple times.
        
        Returns:
            Number of images extracted
        """
        images_count = 0
        image_list = page.get_images(full=True)
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
                    img_bbox = page.get_image_bbox(img)
                    if img_bbox:
                        caption = self._extract_caption_from_page(page, img_bbox)
                except Exception as e:
                    print(f"Error extracting caption for image {img_idx}: {e}")
                
                # Save image
                await self._save_image(
                    image_bytes,
                    document_id,
                    page_number,
                    {"index": img_idx, "source": "pymupdf_embedded", "xref": xref},
                    caption=caption
                )
                images_count += 1
                print(f"Saved embedded image {img_idx} from page {page_number} via PyMuPDF")
            except Exception as e:
                print(f"Error extracting embedded image from PDF page {page_number}: {e}")
                continue
        
        return images_count
    
    async def _extract_bbox_images(
        self,
        page: Any,
        page_number: int,
        document_id: int,
        seen_image_xrefs: Set[int],
        is_table_page: bool
    ) -> int:
        """
        Extract images by rendering their bounding boxes.
        
        This captures images that might be vector graphics or complex figures
        that weren't captured as embedded images. Skips text-heavy pages.
        
        Returns:
            Number of images extracted
        """
        images_count = 0
        
        try:
            page_text = page.get_text()
            text_length = len(page_text.strip()) if page_text else 0
            is_text_heavy = text_length > 1000  # Skip pages with lots of text
            
            if is_text_heavy:
                print(f"Skipping bbox extraction for page {page_number} (text-heavy page)")
                return 0
            
            image_list_full = page.get_images(full=True)
            
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
                            # Render the image region at high resolution (2x zoom)
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat, clip=bbox)
                            
                            # Convert pixmap to PIL Image
                            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # Try to extract caption
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
                            images_count += 1
                            seen_image_xrefs.add(xref)
                            print(f"Saved image from bbox {img_idx} on page {page_number} via PyMuPDF")
                            
                            pix = None  # Free memory
                    except Exception as e:
                        print(f"Could not get bbox for image {xref} on page {page_number}: {e}")
                        continue
                except Exception as e:
                    print(f"Error processing image bbox on page {page_number}: {e}")
                    continue
        except Exception as e:
            print(f"Error checking page text for bbox extraction: {e}")
        
        return images_count
    
    async def _extract_vector_graphics(
        self,
        page: Any,
        page_number: int,
        document_id: int,
        is_table_page: bool
    ) -> int:
        """
        Extract vector graphics and figure regions from page.
        
        Identifies figure regions by clustering vector graphics drawings.
        Only extracts substantial figure regions (>10% but <80% of page).
        Skips text-heavy pages and pages with tables.
        
        Returns:
            Number of images extracted
        """
        images_count = 0
        
        if is_table_page:
            print(f"Skipping vector graphics extraction for page {page_number} (contains tables)")
            return 0
        
        drawings = page.get_drawings()
        if not drawings:
            return 0
        
        print(f"Page {page_number}: Found {len(drawings)} vector graphics")
        
        try:
            page_text = page.get_text()
            text_length = len(page_text.strip()) if page_text else 0
            
            # Only process pages with significant vector graphics
            if len(drawings) > 5:
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
                    
                    # Calculate figure area ratio
                    page_rect = page.rect
                    figure_area_ratio = (figure_bbox.width * figure_bbox.height) / (page_rect.width * page_rect.height)
                    
                    # Determine if this is a substantial figure
                    is_figure_only_page = (len(drawings) > 20 and text_length < 500)
                    is_text_heavy = text_length > 1000
                    is_substantial_figure = ((0.1 < figure_area_ratio < 0.8) or is_figure_only_page) and not is_text_heavy
                    
                    if is_substantial_figure:
                        try:
                            # Render the figure region at high resolution
                            mat = fitz.Matrix(2.0, 2.0)
                            padding = 20
                            clip_rect = fitz.Rect(
                                max(0, figure_bbox.x0 - padding),
                                max(0, figure_bbox.y0 - padding),
                                min(page_rect.width, figure_bbox.x1 + padding),
                                min(page_rect.height, figure_bbox.y1 + padding)
                            )
                            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # Try to extract caption
                            caption = None
                            try:
                                caption = self._extract_caption_from_page(page, clip_rect)
                            except:
                                pass
                            
                            await self._save_image(
                                img_data,
                                document_id,
                                page_number,
                                {
                                    "index": 0,
                                    "source": "pymupdf_figure_region",
                                    "type": "vector_graphics",
                                    "drawings": len(drawings),
                                    "text_chars": text_length,
                                    "bbox": str(clip_rect)
                                },
                                caption=caption
                            )
                            images_count += 1
                            print(f"Saved figure region from page {page_number} ({len(drawings)} drawings)")
                            
                            pix = None
                        except Exception as e:
                            print(f"Error rendering figure region from page {page_number}: {e}")
                    elif is_figure_only_page:
                        # Render entire page as image for figure-only pages
                        try:
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat)
                            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            await self._save_image(
                                img_data,
                                document_id,
                                page_number,
                                {
                                    "index": 0,
                                    "source": "pymupdf_vector",
                                    "type": "vector_graphics_figure",
                                    "drawings": len(drawings),
                                    "text_chars": text_length
                                }
                            )
                            images_count += 1
                            print(f"Saved rendered page {page_number} as image (figure-only page)")
                            
                            pix = None
                        except Exception as e:
                            print(f"Error rendering page {page_number} with vector graphics: {e}")
        except Exception as e:
            print(f"Error analyzing page {page_number} characteristics: {e}")
            import traceback
            traceback.print_exc()
        
        return images_count
    
    def _extract_caption_from_page(self, page: Any, bbox: Any) -> Optional[str]:
        """
        Extract caption text from PDF page near the given bounding box.
        
        Algorithm:
        1. Get all text blocks from page
        2. Find blocks near image horizontally (within tolerance)
        3. Find blocks above or below image (within vertical range)
        4. Look for caption patterns (Figure, Fig., etc.)
        5. Collect consecutive blocks that form the caption
        6. Combine and validate
        
        The algorithm is lenient with gaps (up to 80px) to capture multi-line captions.
        It also checks for continuation patterns (lowercase, punctuation, numbers).
        
        Returns:
            Caption string or None
        """
        try:
            text_blocks = page.get_text("blocks")
            if not text_blocks:
                return None
            
            # Caption patterns to look for
            caption_patterns = ["Figure", "Fig.", "Fig ", "Figure ", "FIGURE", "FIG."]
            
            # Search area: above (within 150px) and below (within 100px) the image
            search_above_y = bbox.y0 + 150
            search_below_y = bbox.y1 + 100
            horizontal_tolerance = 200  # Pixels
            
            # Collect potential caption blocks
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
            caption_blocks = []
            for idx in range(caption_start_idx, len(potential_blocks)):
                y, text, has_pattern = potential_blocks[idx]
                
                if idx == caption_start_idx:
                    # First block - always add
                    caption_blocks.append((y, text))
                else:
                    # Check gap from previous block
                    prev_y = caption_blocks[-1][0]
                    vertical_gap = y - prev_y
                    
                    # Lenient gap tolerance - up to 80 pixels for multi-line captions
                    if vertical_gap < 80:
                        caption_blocks.append((y, text))
                    elif vertical_gap < 150:
                        # Check if this looks like caption continuation
                        text_stripped = text.strip()
                        if text_stripped:
                            first_char = text_stripped[0]
                            if (first_char.islower() or
                                first_char in ',.;:' or
                                first_char.isdigit() or
                                text_stripped.startswith('(') or
                                text_stripped.startswith('[')):
                                caption_blocks.append((y, text))
                            else:
                                # Check if previous caption ended properly
                                prev_text = caption_blocks[-1][1].strip()
                                if prev_text.endswith('.') or prev_text.endswith(':'):
                                    if first_char.islower():
                                        caption_blocks.append((y, text))
                                    else:
                                        break
                                else:
                                    caption_blocks.append((y, text))
                        else:
                            break
                    else:
                        # Gap too large, stop collecting
                        break
            
            if not caption_blocks:
                return None
            
            # Sort by vertical position (top to bottom)
            caption_blocks.sort(key=lambda x: x[0])
            
            # Combine all caption blocks into one caption
            full_caption = " ".join(block_text for _, block_text in caption_blocks)
            
            # Clean up the caption
            full_caption = full_caption.replace('\n', ' ').replace('\r', ' ')
            full_caption = ' '.join(full_caption.split())
            
            # Validate caption - filter out binary data
            if not self.binary_validator.validate_caption(full_caption):
                print(f"WARNING: Caption failed validation (likely binary data), filtering it out")
                return None
            
            # Truncate if too long (max 1000 chars)
            if len(full_caption) > 1000:
                truncated = full_caption[:1000]
                last_period = truncated.rfind('.')
                last_colon = truncated.rfind(':')
                last_semicolon = truncated.rfind(';')
                last_punctuation = max(last_period, last_colon, last_semicolon)
                if last_punctuation > 900:
                    full_caption = truncated[:last_punctuation + 1]
                else:
                    last_space = truncated.rfind(' ')
                    if last_space > 900:
                        full_caption = truncated[:last_space] + "..."
                    else:
                        full_caption = truncated + "..."
            
            return full_caption.strip() if full_caption.strip() else None
            
        except Exception as e:
            print(f"Error in _extract_caption_from_page: {e}")
            return None
    
    def _get_page_number(self, page: Any, default: int) -> int:
        """Extract page number from page object."""
        if hasattr(page, 'page'):
            return page.page
        elif hasattr(page, 'page_number'):
            return page.page_number
        return default
    
    async def _save_image(
        self,
        image_data: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any],
        caption: Optional[str] = None
    ) -> DocumentImage:
        """
        Save an extracted image to filesystem and database.
        
        DONE: Implemented image saving
        - Save image file to disk (PNG format)
        - Create DocumentImage record with metadata
        - Extract caption if available from image_data
        - Validate caption before saving
        
        Args:
            image_data: PIL Image, bytes, or image-like object
            document_id: Document ID for database storage
            page_number: Page number where image was found
            metadata: Additional metadata (source, index, etc.)
            caption: Optional caption text
            
        Returns:
            Created DocumentImage record
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
                caption = self._extract_caption_from_image_data(image_data, metadata)
            
            # Validate caption before saving
            if caption and not self.binary_validator.validate_caption(caption):
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
    
    def _extract_caption_from_image_data(self, image_data: Any, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract caption from image_data object or metadata."""
        caption = None
        
        # Try caption attribute
        if hasattr(image_data, 'caption'):
            try:
                caption_val = image_data.caption
                if callable(caption_val):
                    caption = str(caption_val())
                elif caption_val:
                    caption = str(caption_val)
            except Exception as e:
                print(f"Error extracting caption attribute: {e}")
        
        # Try title attribute
        if not caption and hasattr(image_data, 'title'):
            try:
                title_val = image_data.title
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
        
        # Try metadata
        if not caption and metadata:
            caption = metadata.get('caption') or metadata.get('title')
            if caption:
                caption = str(caption)
        
        return caption

