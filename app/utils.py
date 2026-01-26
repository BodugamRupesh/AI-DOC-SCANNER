import os
import uuid
from pathlib import Path


def generate_doc_id():
    """Generate a unique document ID"""
    return str(uuid.uuid4())[:8]


def ensure_upload_dir():
    """Ensure upload directory exists"""
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def ensure_vector_db_dir():
    """Ensure vector database directory exists"""
    db_dir = Path("./data/vector_db")
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove extra whitespace and newlines
    text = " ".join(text.split())
    return text.strip()
