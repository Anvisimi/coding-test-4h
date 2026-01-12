"""
Table extraction and rendering service.

DONE: Implemented table extraction from Docling
DONE: Implemented structured data extraction (multiple methods)
DONE: Implemented table rendering as images
DONE: Implemented table saving to database
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.document import DocumentTable
from app.core.config import settings
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import fitz  # PyMuPDF


class TableExtractor:
    """
    Extract tables from PDF documents and render them as images.
    
    Uses multiple strategies to extract structured table data:
    1. get_cells() / cells attribute - raw cell structure
    2. export_to_dataframe() - pandas DataFrame (most reliable)
    3. export_to_dict() - dictionary format
    4. .data attribute - direct data access
    5. export_to_markdown() - markdown parsing (fallback)
    
    Renders tables as images using:
    1. Docling's native get_image() method (preferred)
    2. .image attribute (fallback)
    3. PIL rendering from structured data (last resort)
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    async def extract_and_save_tables(
        self,
        doc: Any,
        document_id: int
    ) -> int:
        """
        Extract tables from Docling document and save them.
        
        Args:
            doc: Docling document object
            document_id: Document ID for database storage
            
        Returns:
            Number of tables extracted
        """
        tables_count = 0
        
        # Get tables from document
        tables = []
        try:
            if hasattr(doc, 'tables') and doc.tables:
                tables = list(doc.tables) if not isinstance(doc.tables, list) else doc.tables
                print(f"Found {len(tables)} tables via .tables")
            elif hasattr(doc, 'items') and doc.items:
                tables = [
                    item for item in doc.items
                    if hasattr(item, 'type') and str(getattr(item, 'type', '')).lower() == 'table'
                ]
                print(f"Found {len(tables)} tables via .items")
        except Exception as e:
            print(f"Error getting tables: {e}")
        
        print(f"Total tables found: {len(tables)}")
        
        # Extract and save each table
        for table_idx, table_item in enumerate(tables):
            try:
                page_num = getattr(table_item, 'page', 1) if hasattr(table_item, 'page') else 1
                await self._save_table(
                    table_item,
                    doc,
                    document_id,
                    page_num,
                    {"index": table_idx, "source": "docling"}
                )
                tables_count += 1
            except Exception as e:
                print(f"Error saving table {table_idx}: {e}")
                continue
        
        return tables_count
    
    def identify_pages_with_tables(self, doc: Any, file_path: str) -> set:
        """
        Identify which pages contain tables.
        
        Uses both Docling table detection and PyMuPDF pattern matching
        to identify pages with tables. This helps avoid rendering table
        pages as images.
        
        Returns:
            Set of page numbers (1-indexed) that contain tables
        """
        pages_with_tables = set()
        
        try:
            # Method 1: Get tables from Docling
            if hasattr(doc, 'tables') and doc.tables:
                tables_preview = list(doc.tables) if not isinstance(doc.tables, list) else doc.tables
                for table_item in tables_preview:
                    page_num = getattr(table_item, 'page', None) or getattr(table_item, 'page_number', None)
                    if page_num:
                        pages_with_tables.add(page_num)
                print(f"Pages with tables (from Docling): {sorted(pages_with_tables)}")
            
            # Method 2: Use PyMuPDF to detect table-like structures
            # This helps catch tables that Docling might have missed
            try:
                pdf_doc_temp = fitz.open(file_path)
                for page_num in range(len(pdf_doc_temp)):
                    page = pdf_doc_temp[page_num]
                    page_number = page_num + 1
                    
                    # Check if page has table-like structures
                    blocks = page.get_text("blocks")
                    table_indicators = 0
                    
                    # Look for patterns that suggest tables
                    for block in blocks:
                        block_text = block[4] if len(block) > 4 else ""
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
        
        return pages_with_tables
    
    def _extract_table_data(self, table_data: Any, doc: Any) -> tuple[Optional[List], int, int]:
        """
        Extract structured data from Docling TableItem.
        
        Tries multiple methods in order of preference:
        1. get_cells() / cells - raw cell structure (most accurate)
        2. export_to_dataframe() - pandas DataFrame (most reliable)
        3. export_to_dict() - dictionary format
        4. .data attribute - direct data access
        5. export_to_markdown() - markdown parsing (fallback)
        
        Returns:
            Tuple of (structured_data, rows, columns)
        """
        structured_data = None
        rows = 0
        columns = 0
        
        # Method 0: Try to get raw table structure first (most accurate)
        # This preserves the exact cell structure from the PDF
        try:
            if hasattr(table_data, 'get_cells') or hasattr(table_data, 'cells'):
                cells = None
                if hasattr(table_data, 'get_cells'):
                    cells = table_data.get_cells()
                elif hasattr(table_data, 'cells'):
                    cells = table_data.cells
                
                if cells and isinstance(cells, list) and len(cells) > 0:
                    # Reconstruct table from cells
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
        
        # Method 1: Try export_to_dataframe (most reliable) if pandas is available
        if not structured_data:
            try:
                if hasattr(table_data, 'export_to_dataframe'):
                    try:
                        import pandas as pd
                        # Try with doc argument first (newer API)
                        try:
                            df = table_data.export_to_dataframe(doc)
                        except TypeError:
                            # Fallback if doc argument not supported
                            df = table_data.export_to_dataframe()
                        
                        if df is not None and not df.empty:
                            rows = len(df)
                            columns = len(df.columns)
                            # Convert DataFrame to list of lists for JSON storage
                            # Include column names as first row
                            structured_data = [df.columns.tolist()]
                            # Convert all values to strings, handling NaN values
                            for idx, row in df.iterrows():
                                row_data = []
                                for col in df.columns:
                                    val = row[col]
                                    if pd.isna(val):
                                        row_data.append("")
                                    elif val is None:
                                        row_data.append("")
                                    else:
                                        row_data.append(str(val))
                                structured_data.append(row_data)
                            print(f"Got table via export_to_dataframe: {rows} rows x {columns} columns")
                    except ImportError:
                        print("Pandas not available, trying other methods")
                    except Exception as e:
                        print(f"Error in export_to_dataframe: {e}")
            except Exception as e:
                print(f"Error checking export_to_dataframe: {e}")
        
        # Method 2: Try export_to_dict()
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
        
        # Method 3: Try accessing .data attribute directly
        if rows == 0 or columns == 0 or not structured_data:
            try:
                if hasattr(table_data, 'data'):
                    data_obj = table_data.data
                    
                    # Check if it's a DataFrame-like object
                    if hasattr(data_obj, 'values') and hasattr(data_obj, 'columns'):
                        try:
                            import pandas as pd
                            if isinstance(data_obj, pd.DataFrame):
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
                            pass
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
        
        # Method 4: Try export_to_markdown and parse (last resort)
        if rows == 0 or columns == 0 or not structured_data:
            try:
                if hasattr(table_data, 'export_to_markdown'):
                    markdown = table_data.export_to_markdown()
                    if markdown:
                        print(f"Markdown table length: {len(markdown)} chars")
                        # Parse markdown table
                        lines = [line for line in markdown.split('\n') if line.strip()]
                        parsed_rows = []
                        
                        for line in lines:
                            line = line.strip()
                            # Skip separator lines
                            if line.startswith('|---') or line.startswith('|:---') or line.startswith('|---:') or line.startswith('| ---'):
                                continue
                            
                            # Parse cells - preserve empty cells
                            if '|' in line:
                                parts = line.split('|')
                                cells = []
                                
                                for i, part in enumerate(parts):
                                    part = part.strip()
                                    # Skip leading/trailing empty parts (markdown format)
                                    if i == 0 and part == '':
                                        continue
                                    if i == len(parts) - 1 and part == '':
                                        continue
                                    cells.append(part)
                                
                                if cells:
                                    parsed_rows.append(cells)
                        
                        if parsed_rows:
                            structured_data = parsed_rows
                            rows = len(parsed_rows)
                            if rows > 0:
                                columns = max(len(row) for row in parsed_rows)
                                # Pad rows that are shorter
                                for row in parsed_rows:
                                    while len(row) < columns:
                                        row.append("")
                            
                            non_empty_cells = sum(1 for row in parsed_rows for cell in row if cell and cell.strip())
                            total_cells = rows * columns
                            print(f"Got table via export_to_markdown: {rows} rows x {columns} columns")
                            print(f"Markdown extraction: {non_empty_cells}/{total_cells} non-empty cells")
                        else:
                            # Fallback: store markdown as-is
                            structured_data = {"markdown": markdown, "raw": True}
                            table_lines = [l for l in lines if '|' in l and not (l.startswith('|---') or l.startswith('|:---'))]
                            if table_lines:
                                rows = len(table_lines)
                                first_line = table_lines[0] if table_lines else ""
                                columns = max(first_line.count('|') - 1, 1)
                            print(f"Stored markdown table as raw: {rows} rows x {columns} columns (estimated)")
            except Exception as e:
                print(f"Error in export_to_markdown: {e}")
        
        # Validate and normalize extracted data
        if structured_data:
            structured_data, rows, columns = self._validate_table_data(structured_data, rows, columns)
        
        return structured_data, rows, columns
    
    def _validate_table_data(
        self,
        structured_data: Any,
        rows: int,
        columns: int
    ) -> tuple[List, int, int]:
        """
        Validate and normalize table data structure.
        
        Ensures all rows have the same number of columns and updates
        dimensions to match actual data. Logs data quality metrics.
        
        Returns:
            Tuple of (normalized_data, rows, columns)
        """
        if isinstance(structured_data, list):
            actual_rows = len(structured_data)
            if actual_rows > 0:
                if isinstance(structured_data[0], list):
                    actual_cols = max(len(row) for row in structured_data) if structured_data else 0
                    # Ensure all rows have the same number of columns
                    for row in structured_data:
                        while len(row) < actual_cols:
                            row.append("")
                    
                    rows = actual_rows
                    columns = actual_cols
                    
                    print(f"Final extracted table: {rows} rows x {columns} columns")
                    
                    # Check data quality
                    empty_cells = sum(
                        1 for row in structured_data for cell in row
                        if not cell or (isinstance(cell, str) and cell.strip() == "")
                    )
                    total_cells = sum(len(row) for row in structured_data)
                    non_empty_cells = total_cells - empty_cells
                    
                    print(f"Data quality: {non_empty_cells}/{total_cells} non-empty cells ({100*non_empty_cells/total_cells:.1f}%)")
                    
                    if empty_cells > total_cells * 0.5:
                        print(f"Warning: {empty_cells}/{total_cells} cells are empty - table extraction may be incomplete!")
                else:
                    rows = actual_rows
                    columns = 1
            else:
                rows = 0
                columns = 0
        elif isinstance(structured_data, dict):
            # Keep as-is for dict format
            pass
        else:
            rows = 0
            columns = 0
        
        return structured_data, rows, columns
    
    def _render_table_image(
        self,
        table_data: Any,
        doc: Any,
        structured_data: Optional[List],
        table_image_path: str
    ) -> bool:
        """
        Render table as image.
        
        Tries multiple methods:
        1. Docling's native get_image() method (preferred - most accurate)
        2. .image attribute (fallback)
        3. PIL rendering from structured data (last resort)
        
        Returns:
            True if image was rendered successfully
        """
        table_image_rendered = False
        
        # Method 1: Try Docling's native get_image() method
        try:
            if hasattr(table_data, 'get_image'):
                try:
                    # Try with doc argument first
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
        
        # Method 2: Use provided image attribute
        if not table_image_rendered and hasattr(table_data, 'image') and table_data.image:
            try:
                img = table_data.image
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                elif not isinstance(img, Image.Image):
                    img = Image.open(io.BytesIO(img))
                img.save(table_image_path, "PNG")
                table_image_rendered = True
                print(f"Rendered table using .image attribute")
            except Exception as e:
                print(f"Error using .image attribute: {e}")
        
        # Method 3: Render from structured data (last resort)
        if not table_image_rendered and structured_data and isinstance(structured_data, list) and len(structured_data) > 0:
            try:
                table_image_rendered = self._render_table_from_data(structured_data, table_image_path)
            except Exception as e:
                print(f"Error rendering table from data: {e}")
                import traceback
                traceback.print_exc()
        
        # Create placeholder if all methods failed
        if not table_image_rendered:
            img = Image.new('RGB', (400, 200), color='lightgray')
            img.save(table_image_path, "PNG")
            print(f"Created placeholder image for table")
        
        return table_image_rendered
    
    def _render_table_from_data(self, structured_data: List, table_image_path: str) -> bool:
        """
        Render table image from structured data using PIL.
        
        This is a fallback method that creates a visual representation
        of the table from the extracted structured data. It handles:
        - Dynamic cell widths based on content
        - Multi-line text wrapping
        - Large table scaling
        
        Returns:
            True if rendering was successful
        """
        try:
            # Get dimensions
            rows = len(structured_data)
            if rows == 0:
                return False
            
            if isinstance(structured_data[0], list):
                columns = max(len(row) for row in structured_data) if structured_data else 0
            else:
                columns = 1
            
            # Calculate column widths based on content
            column_max_lengths = [0] * columns
            for row in structured_data:
                if isinstance(row, list):
                    for col_idx, cell in enumerate(row[:columns]):
                        cell_text = str(cell) if cell else ""
                        column_max_lengths[col_idx] = max(column_max_lengths[col_idx], len(cell_text))
            
            # Calculate cell widths (minimum 120px, max 500px)
            font_size = 11
            base_char_width = 7
            min_cell_width = 120
            cell_widths = []
            
            for col_idx, max_len in enumerate(column_max_lengths):
                calculated_width = max(min_cell_width, (max_len * base_char_width) + 40)
                calculated_width = min(calculated_width, 500)
                cell_widths.append(calculated_width)
            
            # Calculate cell height (allow for multi-line text)
            cell_height = 35
            max_lines_per_cell = 1
            for row in structured_data:
                if isinstance(row, list):
                    for col_idx, cell in enumerate(row[:columns]):
                        cell_text = str(cell) if cell else ""
                        if col_idx < len(cell_widths):
                            chars_per_line = (cell_widths[col_idx] - 20) // base_char_width
                            if chars_per_line > 0:
                                lines_needed = (len(cell_text) + chars_per_line - 1) // chars_per_line
                                max_lines_per_cell = max(max_lines_per_cell, lines_needed)
            
            cell_height = max(cell_height, 25 + (max_lines_per_cell - 1) * 15)
            
            # Calculate image dimensions
            img_width = sum(cell_widths) + 20
            img_height = rows * cell_height + 20
            
            # Limit maximum size (scale down if needed)
            max_width = 5000
            max_height = 6000
            if img_width > max_width:
                scale_factor = max_width / img_width
                cell_widths = [int(w * scale_factor) for w in cell_widths]
                img_width = max_width
            if img_height > max_height:
                cell_height = max_height // rows
                img_height = max_height
            
            # Create image and draw table
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to load font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Draw table cells
            y = 10
            for row_idx, row in enumerate(structured_data):
                if y + cell_height > img_height:
                    print(f"Warning: Table too large, stopping at row {row_idx}/{rows}")
                    break
                
                x = 10
                row_data = row if isinstance(row, list) else [str(row)]
                
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
                    
                    # Draw cell text (with wrapping)
                    text_x = x + 5
                    text_y = y + 8
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
                                if len(word) > chars_per_line:
                                    while len(word) > chars_per_line:
                                        lines.append(word[:chars_per_line])
                                        word = word[chars_per_line:]
                                    current_line = word
                                else:
                                    current_line = word
                        
                        if current_line:
                            lines.append(current_line)
                        
                        # Draw each line (max 5 lines per cell)
                        line_height = 14
                        for line_idx, line in enumerate(lines[:5]):
                            if text_y + (line_idx * line_height) + line_height > y + cell_height:
                                break
                            if font:
                                draw.text((text_x, text_y + (line_idx * line_height)), line, fill='black', font=font)
                            else:
                                draw.text((text_x, text_y + (line_idx * line_height)), line, fill='black')
                    else:
                        # Single line text
                        if font:
                            draw.text((text_x, text_y), cell_text, fill='black', font=font)
                        else:
                            draw.text((text_x, text_y), cell_text, fill='black')
                    
                    x += current_cell_width
                y += cell_height
            
            img.save(table_image_path, "PNG")
            print(f"Rendered table image: {rows} rows x {columns} columns, size: {img_width}x{img_height}")
            return True
            
        except Exception as e:
            print(f"Error rendering table image: {e}")
            return False
    
    async def _save_table(
        self,
        table_data: Any,
        doc: Any,
        document_id: int,
        page_number: int,
        metadata: Dict[str, Any]
    ) -> DocumentTable:
        """
        Save an extracted table.
        
        DONE: Implemented table saving
        - Extract structured data using multiple methods
        - Render table as image (prefer Docling native, fallback to PIL)
        - Store structured data as JSON
        - Create DocumentTable record
        - Extract caption if available
        
        Args:
            table_data: Docling TableItem object
            doc: Docling document object (for get_image method)
            document_id: Document ID for database storage
            page_number: Page number where table was found
            metadata: Additional metadata
            
        Returns:
            Created DocumentTable record
        """
        try:
            # Generate unique filename
            table_id = str(uuid.uuid4())
            filename = f"{table_id}.png"
            table_image_path = os.path.join(settings.UPLOAD_DIR, "tables", filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(table_image_path), exist_ok=True)
            
            # Extract structured data
            structured_data, rows, columns = self._extract_table_data(table_data, doc)
            
            # Render table as image
            self._render_table_image(table_data, doc, structured_data, table_image_path)
            
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

