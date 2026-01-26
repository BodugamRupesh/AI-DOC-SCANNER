import streamlit as st
import requests
import os
from pathlib import Path

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📄 RAG Document Chatbot")
st.markdown("Upload PDFs and ask questions using AI-powered RAG (Retrieval-Augmented Generation)")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    backend_status = st.empty()
    
    # Check backend connection
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        backend_status.success("✅ Backend Connected")
    except:
        backend_status.error("❌ Backend Offline")
        st.warning("Backend is not running. Start it with: `uvicorn app.main:app --reload`")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload", "❓ Ask Question", "📋 Manage"])

# ============ TAB 1: UPLOAD ============
with tab1:
    st.header("Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        if st.button("📤 Upload Document", key="upload_btn"):
            with st.spinner("Processing document..."):
                try:
                    # Upload file
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        doc_id = result["doc_id"]
                        
                        # Store doc_id in session
                        st.session_state.current_doc_id = doc_id
                        st.session_state.current_filename = result["filename"]
                        
                        st.success(f"✅ Document uploaded successfully!")
                        st.info(f"""
                        **Document ID:** `{doc_id}`
                        **Filename:** {result['filename']}
                        **Pages:** {result['pages']}
                        """)
                        
                        # Store doc_id for later use
                        if "uploaded_docs" not in st.session_state:
                            st.session_state.uploaded_docs = {}
                        st.session_state.uploaded_docs[doc_id] = {
                            "filename": result["filename"],
                            "pages": result["pages"]
                        }
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============ TAB 2: ASK QUESTION ============
with tab2:
    st.header("Ask Questions")
    
    # Get list of documents
    try:
        response = requests.get(f"{BACKEND_URL}/list", timeout=5)
        if response.status_code == 200:
            docs_list = response.json()["documents"]
            
            if docs_list:
                st.subheader("Select a Document")
                
                doc_options = {doc["filename"]: doc["doc_id"] for doc in docs_list}
                selected_filename = st.selectbox(
                    "Available Documents:",
                    options=list(doc_options.keys()),
                    key="doc_select"
                )
                selected_doc_id = doc_options[selected_filename]
                
                st.write(f"**Selected:** {selected_filename}")
                
                # Question input
                st.subheader("Your Question")
                question = st.text_area(
                    "Ask a question about the document:",
                    placeholder="E.g., What is the main topic of this document?",
                    height=100,
                    key="question_input"
                )
                
                if st.button("🔍 Get Answer", key="ask_btn"):
                    if question.strip():
                        with st.spinner("Searching and generating answer..."):
                            try:
                                payload = {
                                    "question": question,
                                    "doc_id": selected_doc_id
                                }
                                response = requests.post(
                                    f"{BACKEND_URL}/ask",
                                    json=payload,
                                    timeout=60
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    
                                    # Display answer
                                    st.success("✅ Answer Generated")
                                    st.markdown("### Answer")
                                    st.write(result["answer"])
                                    
                                    # Display sources
                                    if result["sources"]:
                                        st.markdown("### 📚 Sources")
                                        for i, source in enumerate(result["sources"], 1):
                                            with st.expander(f"Source {i} - Page {source['page']}"):
                                                st.write(source["text"])
                                    
                                else:
                                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                            
                            except requests.exceptions.ConnectionError:
                                st.error("❌ Cannot connect to backend.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter a question.")
            else:
                st.info("📤 No documents uploaded yet. Go to the 'Upload' tab to upload a PDF.")
    
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")

# ============ TAB 3: MANAGE ============
with tab3:
    st.header("Manage Documents")
    
    # List all documents
    try:
        response = requests.get(f"{BACKEND_URL}/list", timeout=5)
        if response.status_code == 200:
            docs_list = response.json()["documents"]
            total = response.json()["total"]
            
            st.subheader(f"📊 Total Documents: {total}")
            
            if docs_list:
                # Create table
                st.dataframe(
                    [
                        {
                            "Filename": doc["filename"],
                            "Pages": doc["pages"],
                            "Doc ID": doc["doc_id"]
                        }
                        for doc in docs_list
                    ],
                    use_container_width=True
                )
                
                # Delete section
                st.subheader("🗑️ Delete Document")
                doc_to_delete = st.selectbox(
                    "Select document to delete:",
                    options=[doc["filename"] for doc in docs_list],
                    key="delete_select"
                )
                
                if st.button("🗑️ Delete Selected Document", key="delete_btn"):
                    doc_id = next(
                        doc["doc_id"] for doc in docs_list 
                        if doc["filename"] == doc_to_delete
                    )
                    
                    with st.spinner("Deleting document..."):
                        try:
                            response = requests.delete(
                                f"{BACKEND_URL}/delete/{doc_id}",
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                st.success(f"✅ Document deleted: {doc_to_delete}")
                                st.rerun()
                            else:
                                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.info("No documents uploaded yet.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ============ FOOTER ============
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>RAG Document Chatbot • Powered by FastAPI & OpenAI</p>
    </div>
""", unsafe_allow_html=True)
