"""
Patch pgvector.Vector for SQLite compatibility
This must be imported BEFORE any models
"""
import pgvector.sqlalchemy
from sqlalchemy import Text, TypeDecorator


class SQLiteVector(TypeDecorator):
    """SQLite-compatible Vector type"""
    impl = Text
    cache_ok = True
    
    def __init__(self, dimension=None):
        super().__init__()
        self.dimension = dimension
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'sqlite':
            return dialect.type_descriptor(Text())
        return dialect.type_descriptor(pgvector.sqlalchemy.Vector(self.dimension))
    
    def process_bind_param(self, value, dialect):
        if dialect.name == 'sqlite' and value is not None:
            import json
            if isinstance(value, (list, tuple)):
                return json.dumps(value)
            return str(value)
        return value
    
    def process_result_value(self, value, dialect):
        if dialect.name == 'sqlite' and value is not None:
            import json
            try:
                return json.loads(value)
            except:
                return value
        return value


# Replace Vector BEFORE models import it
_original_vector = pgvector.sqlalchemy.Vector
pgvector.sqlalchemy.Vector = SQLiteVector

# Force import models here so they use patched Vector
from app.models import document, conversation  # noqa: F401

