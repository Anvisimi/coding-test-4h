"""
Vector store service using pgvector.

TODO: Implement this service to:
1. Generate embeddings for text chunks
2. Store embeddings in PostgreSQL with pgvector
3. Perform similarity search
4. Link related images and tables
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.document import DocumentChunk, DocumentImage, DocumentTable
from app.core.config import settings
import os


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    
    This is a SKELETON implementation. You need to implement the core logic.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.embeddings_model = None
        self._initialize_embeddings_model()
        self._ensure_extension()
    
    def _initialize_embeddings_model(self):
        """
        Initialize embedding model based on configuration.
        """
        # Try OpenAI first if API key is available
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.embeddings_model = "openai"
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                return
            except Exception as e:
                print(f"Failed to initialize OpenAI: {e}")
        
        # Fallback to HuggingFace sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'  # 384 dimensions, but we'll handle it
            self.embeddings_model = SentenceTransformer(model_name)
            print(f"Using HuggingFace model: {model_name}")
        except Exception as e:
            print(f"Failed to initialize sentence-transformers: {e}")
            raise RuntimeError("No embedding model available. Please install sentence-transformers or provide OpenAI API key.")
    
    def _ensure_extension(self):
        """
        Ensure pgvector extension is enabled.
        
        This is implemented as an example.
        """
        try:
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.db.commit()
        except Exception as e:
            print(f"pgvector extension already exists or error: {e}")
            self.db.rollback()
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        TODO: Implement embedding generation
        - Use OpenAI embeddings API or
        - Use HuggingFace sentence-transformers
        - Return numpy array of embeddings
        
        Example with OpenAI:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
        
        Example with HuggingFace:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
        """
        if self.embeddings_model == "openai":
            # Use OpenAI API
            response = self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        else:
            # Use HuggingFace sentence-transformers
            embedding = self.embeddings_model.encode(text, convert_to_numpy=True)
            # Pad or truncate to match expected dimension (1536 for OpenAI)
            if len(embedding) < settings.EMBEDDING_DIMENSION:
                # Pad with zeros
                padding = np.zeros(settings.EMBEDDING_DIMENSION - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > settings.EMBEDDING_DIMENSION:
                # Truncate
                embedding = embedding[:settings.EMBEDDING_DIMENSION]
            return embedding.astype(np.float32)
    
    async def store_chunk(
        self, 
        content: str, 
        document_id: int,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """
        Store a text chunk with its embedding.
        
        TODO: Implement chunk storage
        1. Generate embedding for content
        2. Create DocumentChunk record
        3. Store in database with embedding
        4. Include metadata (related images, tables, etc.)
        
        Args:
            content: Text content
            document_id: Document ID
            page_number: Page number
            chunk_index: Index of chunk in document
            metadata: Additional metadata (related_images, related_tables, etc.)
            
        Returns:
            Created DocumentChunk
        """
        try:
            # Generate embedding
            embedding = await self.generate_embedding(content)
            
            # Create DocumentChunk record
            chunk = DocumentChunk(
                document_id=document_id,
                content=content,
                embedding=embedding.tolist(),  # Convert to list for pgvector
                page_number=page_number,
                chunk_index=chunk_index,
                extra_metadata=metadata or {}
            )
            
            self.db.add(chunk)
            self.db.commit()
            self.db.refresh(chunk)
            
            return chunk
            
        except Exception as e:
            print(f"Error storing chunk: {e}")
            self.db.rollback()
            raise
    
    async def similarity_search(
        self,
        query: str,
        document_id: Optional[int] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        TODO: Implement similarity search
        1. Generate embedding for query
        2. Use pgvector's cosine similarity (<=> operator)
        3. Filter by document_id if provided
        4. Return top k results with scores
        5. Include related images and tables in results
        
        Example SQL query:
        SELECT 
            id,
            content,
            page_number,
            metadata,
            1 - (embedding <=> :query_embedding) as similarity
        FROM document_chunks
        WHERE document_id = :document_id  -- optional filter
        ORDER BY embedding <=> :query_embedding
        LIMIT :k
        
        Args:
            query: Search query text
            document_id: Optional document ID to filter
            k: Number of results to return
            
        Returns:
            [
                {
                    "content": "...",
                    "score": 0.95,
                    "page_number": 3,
                    "metadata": {...},
                    "related_images": [...],
                    "related_tables": [...]
                }
            ]
        """
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Use ORM query with pgvector operations
            # Convert embedding to list for pgvector
            embedding_list = query_embedding.tolist()
            
            # Build base query
            query = self.db.query(
                DocumentChunk.id,
                DocumentChunk.content,
                DocumentChunk.page_number,
                DocumentChunk.chunk_index,
                DocumentChunk.extra_metadata
            )
            
            # Filter by document if provided
            if document_id:
                query = query.filter(DocumentChunk.document_id == document_id)
            
            # Use raw SQL for pgvector similarity search since ORM doesn't support <=> operator directly
            # Convert embedding to string format for PostgreSQL
            embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
            
            if document_id:
                sql = text("""
                    SELECT 
                        id,
                        content,
                        page_number,
                        chunk_index,
                        extra_metadata,
                        1 - (embedding <=> CAST(:embedding_vec AS vector)) as similarity
                    FROM document_chunks
                    WHERE document_id = :doc_id
                    ORDER BY embedding <=> CAST(:embedding_vec AS vector)
                    LIMIT :limit_k
                """)
                result = self.db.execute(
                    sql,
                    {
                        "embedding_vec": embedding_str,
                        "doc_id": document_id,
                        "limit_k": k
                    }
                )
            else:
                sql = text("""
                    SELECT 
                        id,
                        content,
                        page_number,
                        chunk_index,
                        extra_metadata,
                        1 - (embedding <=> CAST(:embedding_vec AS vector)) as similarity
                    FROM document_chunks
                    ORDER BY embedding <=> CAST(:embedding_vec AS vector)
                    LIMIT :limit_k
                """)
                result = self.db.execute(
                    sql,
                    {
                        "embedding_vec": embedding_str,
                        "limit_k": k
                    }
                )
            
            # Process results
            results = []
            for row in result:
                chunk_id = row[0]
                content = row[1]
                page_number = row[2]
                chunk_index = row[3]
                metadata = row[4] or {}
                similarity = float(row[5])
                
                # Get related images and tables
                related_images = []
                related_tables = []
                
                # Check metadata for related content IDs
                if metadata:
                    image_ids = metadata.get("related_images", [])
                    table_ids = metadata.get("related_tables", [])
                    
                    if image_ids:
                        images = self.db.query(DocumentImage).filter(
                            DocumentImage.id.in_(image_ids)
                        ).all()
                        for img in images:
                            related_images.append({
                                "id": img.id,
                                "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                                "caption": img.caption,
                                "page": img.page_number
                            })
                    
                    if table_ids:
                        tables = self.db.query(DocumentTable).filter(
                            DocumentTable.id.in_(table_ids)
                        ).all()
                        for tbl in tables:
                            related_tables.append({
                                "id": tbl.id,
                                "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                                "caption": tbl.caption,
                                "page": tbl.page_number,
                                "data": tbl.data
                            })
                
                # Also find images/tables on the same page
                if page_number:
                    page_images = self.db.query(DocumentImage).filter(
                        DocumentImage.document_id == (document_id or DocumentImage.document_id),
                        DocumentImage.page_number == page_number
                    ).limit(3).all()
                    
                    for img in page_images:
                        if not any(r["id"] == img.id for r in related_images):
                            related_images.append({
                                "id": img.id,
                                "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                                "caption": img.caption,
                                "page": img.page_number
                            })
                    
                    page_tables = self.db.query(DocumentTable).filter(
                        DocumentTable.document_id == (document_id or DocumentTable.document_id),
                        DocumentTable.page_number == page_number
                    ).limit(3).all()
                    
                    for tbl in page_tables:
                        if not any(r["id"] == tbl.id for r in related_tables):
                            related_tables.append({
                                "id": tbl.id,
                                "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                                "caption": tbl.caption,
                                "page": tbl.page_number,
                                "data": tbl.data
                            })
                
                results.append({
                    "id": chunk_id,
                    "content": content,
                    "score": similarity,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "metadata": metadata,
                    "related_images": related_images,
                    "related_tables": related_tables
                })
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            raise
    
    async def get_related_content(
        self,
        chunk_ids: List[int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get related images and tables for given chunks.
        
        TODO: Implement related content retrieval
        - Query DocumentImage and DocumentTable based on metadata
        - Return organized by type (images, tables)
        
        Returns:
            {
                "images": [...],
                "tables": [...]
            }
        """
        if not chunk_ids:
            return {"images": [], "tables": []}
        
        # Get chunks
        chunks = self.db.query(DocumentChunk).filter(
            DocumentChunk.id.in_(chunk_ids)
        ).all()
        
        image_ids = set()
        table_ids = set()
        page_numbers = set()
        document_ids = set()
        
        # Extract related content IDs and page numbers from metadata
        for chunk in chunks:
            if chunk.extra_metadata:
                image_ids.update(chunk.extra_metadata.get("related_images", []))
                table_ids.update(chunk.extra_metadata.get("related_tables", []))
            if chunk.page_number:
                page_numbers.add(chunk.page_number)
            if chunk.document_id:
                document_ids.add(chunk.document_id)
        
        # Get images
        images = []
        if image_ids:
            image_records = self.db.query(DocumentImage).filter(
                DocumentImage.id.in_(list(image_ids))
            ).all()
            for img in image_records:
                images.append({
                    "id": img.id,
                    "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                    "caption": img.caption,
                    "page": img.page_number,
                    "width": img.width,
                    "height": img.height
                })
        
        # Also get images from same pages
        if page_numbers and document_ids:
            page_images = self.db.query(DocumentImage).filter(
                DocumentImage.document_id.in_(list(document_ids)),
                DocumentImage.page_number.in_(list(page_numbers))
            ).all()
            for img in page_images:
                if not any(i["id"] == img.id for i in images):
                    images.append({
                        "id": img.id,
                        "url": f"/uploads/images/{os.path.basename(img.file_path)}",
                        "caption": img.caption,
                        "page": img.page_number,
                        "width": img.width,
                        "height": img.height
                    })
        
        # Get tables
        tables = []
        if table_ids:
            table_records = self.db.query(DocumentTable).filter(
                DocumentTable.id.in_(list(table_ids))
            ).all()
            for tbl in table_records:
                tables.append({
                    "id": tbl.id,
                    "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                    "caption": tbl.caption,
                    "page": tbl.page_number,
                    "data": tbl.data,
                    "rows": tbl.rows,
                    "columns": tbl.columns
                })
        
        # Also get tables from same pages
        if page_numbers and document_ids:
            page_tables = self.db.query(DocumentTable).filter(
                DocumentTable.document_id.in_(list(document_ids)),
                DocumentTable.page_number.in_(list(page_numbers))
            ).all()
            for tbl in page_tables:
                if not any(t["id"] == tbl.id for t in tables):
                    tables.append({
                        "id": tbl.id,
                        "url": f"/uploads/tables/{os.path.basename(tbl.image_path)}",
                        "caption": tbl.caption,
                        "page": tbl.page_number,
                        "data": tbl.data,
                        "rows": tbl.rows,
                        "columns": tbl.columns
                    })
        
        return {
            "images": images,
            "tables": tables
        }
