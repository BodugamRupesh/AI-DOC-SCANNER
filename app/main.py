from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
from dotenv import load_dotenv

from app.schemas import UploadResponse, AskRequest, AskResponse, DeleteResponse, ListResponse, Source
from app.pdf_loader import PDFLoader
from app.vector_store import VectorStore
from app.rag_pipeline import RAGPipeline
from app.utils import generate_doc_id, ensure_upload_dir

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RAG Document Chatbot API",
    description="Upload PDFs and ask questions using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStore()
rag_pipeline = RAGPipeline(api_key=os.getenv("GROQ_API_KEY"))
upload_dir = ensure_upload_dir()

# In-memory storage for document metadata (in production, use a database)
documents_db = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "RAG Document Chatbot API is running",
        "endpoints": ["/upload", "/ask", "/delete", "/list", "/docs"]
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it
    
    - Extracts text
    - Creates embeddings
    - Stores in vector database
    """
    try:
        # Validate file
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Generate document ID
        doc_id = generate_doc_id()
        
        # Save file
        file_path = upload_dir / f"{doc_id}_{file.filename}"
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Extract text with page tracking
        pages_text = PDFLoader.extract_text_with_pages(str(file_path))
        full_text = "\n\n".join(pages_text.values())
        page_count = len(pages_text)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Add to vector store
        vector_store.add_documents(
            doc_id=doc_id,
            text=full_text,
            filename=file.filename,
            pages_text=pages_text
        )
        
        # Store metadata
        documents_db[doc_id] = {
            "filename": file.filename,
            "pages": page_count,
            "file_path": str(file_path)
        }
        
        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            pages=page_count,
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about uploaded documents
    
    - Searches vector store for relevant content
    - Generates answer using LLM
    - Returns answer with citations
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        doc_id = request.doc_id
        if not doc_id or doc_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Search vector store
        search_results = vector_store.search(doc_id, request.question, top_k=3)
        
        if not search_results:
            return AskResponse(
                answer="I couldn't find relevant information in the document.",
                sources=[],
                doc_id=doc_id
            )
        
        # Generate answer using RAG
        answer, sources = rag_pipeline.process_question(request.question, search_results)
        
        return AskResponse(
            answer=answer,
            sources=[Source(page=s["page"], text=s["text"]) for s in sources],
            doc_id=doc_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.delete("/delete/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """
    Delete a document from vector store and storage
    """
    try:
        if doc_id not in documents_db:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store
        vector_store.delete_collection(doc_id)
        
        # Delete file
        file_info = documents_db[doc_id]
        if Path(file_info["file_path"]).exists():
            Path(file_info["file_path"]).unlink()
        
        # Remove metadata
        del documents_db[doc_id]
        
        return DeleteResponse(
            status="success",
            message=f"Document {file_info['filename']} deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/list", response_model=ListResponse)
async def list_documents():
    """
    List all uploaded documents
    """
    documents = [
        {
            "doc_id": doc_id,
            "filename": info["filename"],
            "pages": info["pages"]
        }
        for doc_id, info in documents_db.items()
    ]
    
    return ListResponse(
        documents=documents,
        total=len(documents)
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment"""
    return {"status": "healthy", "service": "RAG Document Chatbot API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
