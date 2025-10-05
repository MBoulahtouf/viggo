"""
Main Streamlit application for Viggo - Intelligent Reading Assistant.

Simplified two-tab interface: Chat (upload + query) and Graph (Neo4j visualization).
"""

import streamlit as st
import sys
import os
from typing import Dict, Any, Optional

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.api_client import api_client
from utils.session_manager import session_manager

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Initialize session state
session_manager.initialize_session()


def render_document_upload():
    """Render document upload interface."""
    st.markdown("### Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=['pdf', 'epub'],
        help="Supported formats: PDF, EPUB. Maximum file size: 50MB",
        key="main_uploader"
    )
    
    if uploaded_file is not None:
        # Show file details (responsive)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Filename", uploaded_file.name)
        
        with col2:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("Size", f"{file_size_mb:.1f} MB")
        
        with col3:
            st.metric("Type", uploaded_file.type or "Unknown")
        
        # Upload button
        if st.button("Upload & Process Document", type="primary", use_container_width=True):
            with st.spinner("Uploading and processing your document..."):
                try:
                    # Upload document
                    file_content = uploaded_file.getvalue()
                    response = api_client.upload_document(file_content, uploaded_file.name)
                    
                    if "error" in response:
                        st.error(f"Upload failed: {response['error']}")
                        return False
                    
                    # Update session state
                    session_manager.set_current_document(uploaded_file.name)
                    session_manager.set_document_info(response)
                    
                    st.success("Document uploaded and processed successfully!")
                    
                    # Display processing results - handle new API response structure
                    if "data" in response and "document_info" in response["data"]:
                        data = response["data"]
                        doc_info = data["document_info"]
                        st.info(f"**{doc_info.get('filename', uploaded_file.name)}** processed")
                        st.info(f"**{data.get('num_chunks_indexed', 0)}** chunks created")
                        st.info(f"**{doc_info.get('total_pages', 'Unknown')}** pages detected")
                    elif "document_info" in response:
                        # Fallback for old response structure
                        doc_info = response["document_info"]
                        st.info(f"**{doc_info.get('filename', uploaded_file.name)}** processed")
                        st.info(f"**{response.get('num_chunks_indexed', 0)}** chunks created")
                        st.info(f"**{doc_info.get('total_pages', 'Unknown')}** pages detected")
                    
                    # Auto-refresh to show query interface
                    st.rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
                    return False
    
    return False


def render_document_status():
    """Render current document status."""
    current_doc = session_manager.get_current_document()
    document_info = session_manager.get_document_info()
    progress = session_manager.get_reading_stats()
    
    if current_doc and document_info:
        st.markdown("### Current Document")
        
        # Document info in columns (responsive)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("Document", current_doc.split('/')[-1] if '/' in current_doc else current_doc)
        
        with col2:
            # Handle new API response structure
            if "data" in document_info and "document_info" in document_info["data"]:
                total_pages = document_info["data"]["document_info"].get("total_pages", "Unknown")
                st.metric("Total Pages", total_pages)
            elif "document_info" in document_info:
                # Fallback for old response structure
                total_pages = document_info["document_info"].get("total_pages", "Unknown")
                st.metric("Total Pages", total_pages)
            else:
                st.metric("Total Pages", "Unknown")
        
        with col3:
            current_page = progress['current_page']
            if current_page == 0:
                st.metric("Current Page", "Not Started")
            else:
                st.metric("Current Page", current_page)
        
        with col4:
            if current_page == 0:
                st.metric("Progress", "0.0%")
            else:
                st.metric("Progress", f"{progress['progress_percentage']:.1f}%")
        
        # Reading status and spoiler protection
        if current_page == 0:
            st.info("**📖 Ready to Start Reading** - Set your progress below to begin asking questions")
        elif progress['spoiler_protection']:
            st.success(f"**Spoiler Protection ENABLED** - Queries limited to pages 1-{progress['current_page']}")
        else:
            st.warning("**Spoiler Protection DISABLED** - You can ask about any part of the book")
        
        # Show note about page count if it seems incorrect
        total_pages = 0
        if "data" in document_info and "document_info" in document_info["data"]:
            total_pages = document_info["data"]["document_info"].get("total_pages", 0)
        elif "document_info" in document_info:
            total_pages = document_info["document_info"].get("total_pages", 0)
        
        if total_pages < 10:  # Suspiciously low page count
            st.info("**Note**: If you're having trouble with queries, try re-uploading your document to ensure proper indexing.")
        
        # Reading progress controls
        st.markdown("#### 📖 Reading Progress")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("**Set Current Page:**")
            new_page = st.number_input(
                "Page:", 
                min_value=0, 
                max_value=total_pages if total_pages > 0 else 1000,
                value=progress['current_page'],
                key="page_input"
            )
            if st.button("Update Page", use_container_width=True):
                session_manager.update_reading_progress(new_page)
                st.rerun()
        
        with col2:
            st.markdown("**Quick Actions:**")
            if st.button("Start Reading (Page 1)", use_container_width=True):
                session_manager.update_reading_progress(1)
                st.rerun()
            
            if st.button("Mark as Finished", use_container_width=True):
                session_manager.mark_as_finished()
                st.rerun()
        
        with col3:
            st.markdown("**Spoiler Protection:**")
            spoiler_enabled = progress.get('spoiler_protection', True)
            new_spoiler_setting = st.checkbox(
                "Enable Spoiler Protection", 
                value=spoiler_enabled,
                help="When enabled, queries are limited to pages you've already read"
            )
            if new_spoiler_setting != spoiler_enabled:
                session_manager.toggle_spoiler_protection()
                st.rerun()
        
        # Document management
        st.markdown("#### 📄 Document Management")
        if st.button("Change Document", use_container_width=True):
            session_manager.reset_session()
            st.rerun()
        
        return True
    
    return False


def render_query_interface():
    """Render the query interface."""
    progress = session_manager.get_reading_stats()
    current_page = progress['current_page']
    is_finished = progress['is_finished']
    
    # Check if user has made progress or marked as finished
    if current_page == 0 and not is_finished:
        st.markdown("### Ask Questions About Your Book")
        st.info("📖 **Please set your reading progress first!** Use the controls above to set your current page or mark the book as finished.")
        return
    
    st.markdown("### Ask Questions About Your Book")
    
    # Show reading status
    if is_finished:
        st.success("✅ **Book Finished** - You can ask questions about any part of the book!")
    elif current_page > 0:
        st.info(f"📖 **Reading Progress** - You can ask questions about pages 1-{current_page}")
    
    # Query input
    query = st.text_area(
        "Ask a question about your book:",
        placeholder="e.g., Who is the main character? What happens in chapter 3? Explain the relationship between...",
        height=120,
        max_chars=config.MAX_QUERY_LENGTH,
        help=f"Ask questions about characters, plot, themes, or any aspect of the book. Maximum {config.MAX_QUERY_LENGTH} characters.",
        key="main_query_input"
    )
    
    # Submit button
    if st.button("Ask Question", type="primary", disabled=not query.strip(), use_container_width=True):
        return query.strip()
    
    return None


def render_query_response(response: Dict[str, Any], query: str):
    """Render the query response."""
    st.markdown("#### Answer")
    
    if "error" in response:
        error_msg = response['error']
        st.error(f"**Query Failed**: {error_msg}")
        return
    
    # Extract response data
    if "data" in response:
        data = response["data"]
    else:
        data = response
    
    # Display answer
    answer = data.get("answer", "No answer provided")
    st.markdown(f"**Answer:** {answer}")
    
    # Display source pages
    source_pages = data.get("source_pages", [])
    if source_pages:
        st.markdown("#### Source Pages")
        
        for i, page in enumerate(source_pages, 1):
            with st.expander(f"Source {i}: Page {page.get('page_number', 'Unknown')}"):
                st.write(f"**Content:** {page.get('content', 'No content available')}")
                st.write(f"**Relevance Score:** {page.get('relevance_score', 0):.2f}")
    
    # Display source information
    if source_pages:
        st.markdown("#### Source Information")
        st.metric("Source Pages", len(source_pages))
    
    # Add to query history
    session_manager.add_query_to_history(
        query, 
        answer, 
        session_manager.get_user_progress().get("current_page")
    )


def render_graph_visualization():
    """Render Neo4j graph visualization."""
    st.markdown("### Knowledge Graph")
    
    # Check if document is loaded
    current_doc = session_manager.get_current_document()
    if not current_doc:
        st.info("Please upload a document first to view the knowledge graph.")
        return
    
    # Graph visualization options (responsive)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        graph_type = st.selectbox(
            "Graph Type:",
            ["Entities", "Relationships", "Full Graph"],
            help="Choose what to visualize in the graph"
        )
    
    with col2:
        max_nodes = st.slider("Max Nodes:", min_value=10, max_value=100, value=50)
    
    # Fetch graph data
    if st.button("Refresh Graph", type="primary"):
        with st.spinner("Loading graph data..."):
            try:
                # This would call your graph API endpoint
                # For now, we'll show a placeholder
                st.info("Graph visualization would be implemented here")
                st.info("This would connect to your Neo4j graph service")
                
                # Placeholder graph data
                st.markdown("#### Sample Graph Structure")
                st.json({
                    "nodes": [
                        {"id": "character_1", "label": "Main Character", "type": "Character"},
                        {"id": "location_1", "label": "Story Location", "type": "Location"},
                        {"id": "event_1", "label": "Key Event", "type": "Event"}
                    ],
                    "edges": [
                        {"source": "character_1", "target": "location_1", "relationship": "lives_in"},
                        {"source": "character_1", "target": "event_1", "relationship": "participates_in"}
                    ]
                })
                
            except Exception as e:
                st.error(f"Failed to load graph: {str(e)}")


def main():
    """Main application with two tabs."""
    # Custom CSS for clean typography and responsive layout
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main container responsive sizing */
    .main .block-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 1000px !important;
        padding: 1rem 1rem 1rem 1rem !important;
    }
    
    /* Responsive layout for different screen sizes */
    @media (max-width: 1200px) {
        .main .block-container {
            max-width: 900px !important;
        }
    }
    
    @media (max-width: 992px) {
        .main .block-container {
            max-width: 800px !important;
        }
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            max-width: 100% !important;
            padding: 0.5rem !important;
        }
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
    }
    
    .stButton > button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stSelectbox > div > div > select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stMetric {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stAlert {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar responsive adjustments */
    .css-1d391kg {
        width: 20rem !important;
    }
    
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 15rem !important;
        }
    }
    
    /* File uploader responsive */
    .stFileUploader > div {
        width: 100% !important;
    }
    
    /* Columns responsive behavior */
    .stColumns > div {
        min-width: 0 !important;
    }
    
    /* Metrics responsive */
    .stMetric > div {
        min-width: 0 !important;
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Viggo - Intelligent Reading Assistant")
    st.markdown("Upload documents and explore knowledge graphs")
    
    # Check API connection
    with st.sidebar:
        st.markdown("### System Status")
        with st.spinner("Checking API..."):
            try:
                health_response = api_client.health_check()
                if "error" not in health_response:
                    st.success("API Connected")
                else:
                    st.error("API Error")
            except Exception as e:
                st.error("Connection Failed")
    
    # Main tabs
    tab1, tab2 = st.tabs(["Chat", "Graph"])
    
    with tab1:
        # Chat tab - combines upload and query
        current_doc = session_manager.get_current_document()
        
        if not current_doc:
            # No document loaded - show upload interface
            st.markdown("### Get Started")
            st.info("**No document loaded** - Upload a document to get started!")
            
            render_document_upload()
            
        else:
            # Document loaded - show status and query interface
            if render_document_status():
                # Show query interface
                query = render_query_interface()
                
                if query:
                    with st.spinner("Processing your question..."):
                        response = api_client.query_document(
                            query, 
                            session_manager.get_user_progress().get("current_page")
                        )
                        
                        render_query_response(response, query)
    
    with tab2:
        # Graph tab - Neo4j visualization
        render_graph_visualization()


if __name__ == "__main__":
    main()
