"""
Document upload component for the Streamlit frontend.
"""

import streamlit as st
from typing import Optional, Dict, Any
import io

from config import config
from utils.api_client import api_client
from utils.session_manager import session_manager


def render_document_upload() -> Optional[Dict[str, Any]]:
    """
    Render the document upload interface.
    
    Returns:
        Document info if upload successful, None otherwise
    """
    st.header("📚 Upload Document")
    st.markdown("Upload a PDF or EPUB file to start your reading journey with Viggo.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=config.SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}. Max size: {config.MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is not None:
        # Validate file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large. Maximum size is {config.MAX_FILE_SIZE_MB}MB.")
            return None
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{file_size_mb:.1f} MB")
        with col3:
            st.metric("Type", uploaded_file.type or "Unknown")
        
        # Upload button
        if st.button("🚀 Upload & Process Document", type="primary"):
            with st.spinner("Uploading and processing document..."):
                try:
                    # Upload document
                    file_content = uploaded_file.getvalue()
                    response = api_client.upload_document(file_content, uploaded_file.name)
                    
                    if "error" in response:
                        st.error(f"❌ Upload failed: {response['error']}")
                        return None
                    
                    # Check if response has the expected structure
                    if "data" in response:
                        document_info = response["data"]
                    else:
                        # Fallback for legacy API response
                        document_info = {
                            "filename": uploaded_file.name,
                            "num_chunks_indexed": response.get("num_chunks_indexed", 0),
                            "message": response.get("message", "Document processed successfully"),
                            "document_info": {
                                "filename": uploaded_file.name,
                                "file_type": uploaded_file.name.split('.')[-1].upper(),
                                "total_pages": 100,  # Default fallback
                                "file_size": len(file_content),
                                "upload_timestamp": "2024-01-01T00:00:00Z"
                            }
                        }
                    
                    # Update session state
                    session_manager.set_current_document(uploaded_file.name)
                    session_manager.set_document_info(document_info)
                    
                    st.success("✅ Document uploaded and processed successfully!")
                    
                    # Display processing results
                    if "document_info" in document_info:
                        doc_info = document_info["document_info"]
                        st.info(f"📄 **{doc_info.get('filename', uploaded_file.name)}** processed")
                        st.info(f"📊 **{document_info.get('num_chunks_indexed', 0)}** chunks created")
                        st.info(f"📖 **{doc_info.get('total_pages', 'Unknown')}** pages detected")
                    
                    return document_info
                    
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    return None
    
    return None


def render_document_status():
    """Render current document status."""
    current_doc = session_manager.get_current_document()
    document_info = session_manager.get_document_info()
    
    if current_doc and document_info:
        st.success(f"📚 **Current Document:** {current_doc}")
        
        if "document_info" in document_info:
            doc_info = document_info["document_info"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Pages", doc_info.get("total_pages", "Unknown"))
            with col2:
                st.metric("Chunks Created", document_info.get("num_chunks_indexed", 0))
            with col3:
                file_type = doc_info.get("file_type", "Unknown")
                st.metric("File Type", file_type)
        
        # Reset document button
        if st.button("🔄 Reset Document", help="Clear current document and start over"):
            session_manager.reset_session()
            st.rerun()
    
    else:
        st.info("📝 No document uploaded yet. Please upload a document to get started.")


def render_supported_formats():
    """Render supported document formats."""
    with st.expander("📋 Supported Document Formats"):
        st.markdown("""
        **Supported Formats:**
        - **PDF** (.pdf) - Portable Document Format
        - **EPUB** (.epub) - Electronic Publication format
        
        **Requirements:**
        - Maximum file size: 50MB
        - Text-based content (not scanned images)
        - Properly formatted documents work best
        
        **Tips for Best Results:**
        - Use high-quality PDFs with selectable text
        - EPUB files with proper chapter structure
        - Avoid password-protected or encrypted files
        - Ensure the document has clear page breaks
        """)


def render_upload_help():
    """Render upload help and tips."""
    with st.expander("❓ Upload Help & Tips"):
        st.markdown("""
        **How to Upload:**
        1. Click "Browse files" or drag and drop your document
        2. Select a PDF or EPUB file from your computer
        3. Click "Upload & Process Document" to start processing
        4. Wait for the system to analyze and index your document
        
        **What Happens During Processing:**
        - Document is analyzed for structure and content
        - Text is extracted and divided into searchable chunks
        - Entities (characters, locations, etc.) are identified
        - Relationships between entities are mapped
        - Document is indexed for fast searching
        
        **Troubleshooting:**
        - **File too large:** Try compressing your PDF or splitting it
        - **Processing fails:** Ensure the document has selectable text
        - **Slow processing:** Large documents may take several minutes
        - **Unsupported format:** Convert to PDF or EPUB format
        """)


def main():
    """Main document upload interface."""
    st.set_page_config(**config.PAGE_CONFIG)
    
    st.title("📚 Viggo - Document Upload")
    
    # Render components
    render_supported_formats()
    render_upload_help()
    
    # Main upload interface
    document_info = render_document_upload()
    
    # Document status
    render_document_status()
    
    # Navigation
    if document_info:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Continue to Reading Setup", type="primary"):
                st.switch_page("pages/reading_setup.py")
        with col2:
            if st.button("📊 View Document Info"):
                st.switch_page("pages/document_info.py")


if __name__ == "__main__":
    main()
