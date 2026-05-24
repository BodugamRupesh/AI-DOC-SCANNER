import os
import json
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from app.utils import ensure_vector_db_dir, chunk_text
import numpy as np


class SimpleVectorStore:
    """Simple in-memory vector store using cloud embeddings (works with Python 3.14)"""
    
    def __init__(self):
        ensure_vector_db_dir()
        self.stores = {}  # doc_id -> {embeddings, documents, metadata}
        self.db_path = Path("./data/vector_db")
        
        # Detect active API key
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if gemini_key:
            self.client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/"
            )
            self.model = "text-embedding-004"
        elif openai_key:
            self.client = OpenAI(api_key=openai_key)
            self.model = "text-embedding-3-small"
        elif groq_key:
            self.client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.model = "nomic-embed-text"
        else:
            raise ValueError(
                "No API key found for vector store. Please set GEMINI_API_KEY, "
                "OPENAI_API_KEY, or GROQ_API_KEY in your environment."
            )
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using the configured API client"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings via API: {str(e)}")
            
    def get_or_create_collection(self, doc_id: str):
        """Get or create a collection for a document"""
        if doc_id not in self.stores:
            self.stores[doc_id] = {
                "embeddings": np.array([]),
                "documents": [],
                "metadata": []
            }
            # Try to load existing data if present
            self._load_from_disk(doc_id)
        return doc_id
    
    def add_documents(self, doc_id: str, text: str, filename: str, pages_text: Dict[int, str]):
        """
        Add document to vector store with page tracking
        """
        self.get_or_create_collection(doc_id)
        
        # Create chunks with page tracking
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        embeddings = self._get_embeddings(chunks)
        
        # Try to map chunks to pages (approximate)
        metadatas = []
        chunk_char_pos = 0
        page_num = 1
        
        for chunk in chunks:
            # Find which page this chunk belongs to
            for p_num in sorted(pages_text.keys()):
                if chunk_char_pos < len("\n\n".join(
                    pages_text[p] for p in sorted(pages_text.keys()) if p <= p_num
                )):
                    page_num = p_num
                    break
            
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num
            })
            chunk_char_pos += len(chunk)
        
        # Store in memory
        self.stores[doc_id]["embeddings"] = np.array(embeddings)
        self.stores[doc_id]["documents"] = chunks
        self.stores[doc_id]["metadata"] = metadatas
        
        # Also save to disk for persistence
        self._save_to_disk(doc_id)
    
    def search(self, doc_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant chunks
        """
        if doc_id not in self.stores:
            return []
        
        store = self.stores[doc_id]
        
        if len(store["documents"]) == 0:
            return []
        
        # Encode query
        query_embedding = self._get_embeddings([query])[0]
        
        # Calculate similarities (cosine)
        embeddings = store["embeddings"]
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                formatted_results.append({
                    "text": store["documents"][idx],
                    "page": store["metadata"][idx]["page"],
                    "distance": float(1 - similarities[idx])  # Convert similarity to distance
                })
        
        return formatted_results
    
    def delete_collection(self, doc_id: str):
        """Delete a document collection"""
        if doc_id in self.stores:
            del self.stores[doc_id]
            # Remove from disk
            db_file = self.db_path / f"{doc_id}.json"
            if db_file.exists():
                db_file.unlink()
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        return list(self.stores.keys())
    
    def get_collection_stats(self, doc_id: str) -> Dict:
        """Get statistics for a collection"""
        if doc_id not in self.stores:
            return None
        return {
            "doc_id": doc_id,
            "documents": len(self.stores[doc_id]["documents"])
        }
    
    def _save_to_disk(self, doc_id: str):
        """Save store to disk for persistence"""
        try:
            store = self.stores[doc_id]
            data = {
                "embeddings": store["embeddings"].tolist(),
                "documents": store["documents"],
                "metadata": store["metadata"]
            }
            
            db_file = self.db_path / f"{doc_id}.json"
            with open(db_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save to disk: {e}")
    
    def _load_from_disk(self, doc_id: str):
        """Load store from disk"""
        try:
            db_file = self.db_path / f"{doc_id}.json"
            if db_file.exists():
                with open(db_file, 'r') as f:
                    data = json.load(f)
                
                self.stores[doc_id] = {
                    "embeddings": np.array(data["embeddings"]),
                    "documents": data["documents"],
                    "metadata": data["metadata"]
                }
        except Exception as e:
            print(f"Warning: Could not load from disk: {e}")


# Alias for compatibility
VectorStore = SimpleVectorStore
