"""
Document information page for the Viggo Streamlit frontend.
"""

import streamlit as st
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.session_manager import session_manager
from utils.api_client import api_client

# Initialize session
session_manager.initialize_session()

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

st.title("📊 Document Information")

# Check if document is loaded
if not session_manager.get_current_document():
    st.error("❌ No document loaded. Please upload a document first.")
    if st.button("📚 Go to Document Upload"):
        st.switch_page("pages/document_upload.py")
    st.stop()

# Get document information
document_info = session_manager.get_document_info()
progress = session_manager.get_reading_stats()

if not document_info:
    st.error("❌ No document information available.")
    st.stop()

# Display document overview
st.header("📚 Document Overview")

doc_info = document_info.get("document_info", {})
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Filename", doc_info.get("filename", "Unknown"))

with col2:
    st.metric("File Type", doc_info.get("file_type", "Unknown"))

with col3:
    st.metric("Total Pages", doc_info.get("total_pages", 0))

with col4:
    file_size = doc_info.get("file_size", 0)
    st.metric("File Size", f"{file_size / (1024*1024):.1f} MB")

# Processing information
st.header("⚙️ Processing Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Chunks Created", document_info.get("num_chunks_indexed", 0))

with col2:
    st.metric("Processing Time", f"{document_info.get('processing_time', 0):.2f}s")

with col3:
    upload_time = doc_info.get("upload_timestamp", "Unknown")
    if upload_time != "Unknown":
        upload_time = upload_time[:19]  # Remove timezone info
    st.metric("Upload Time", upload_time)

# Reading progress
st.header("📖 Reading Progress")

progress_percentage = progress["progress_percentage"]
remaining_pages = max(0, progress["total_pages"] - progress["current_page"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Page", progress["current_page"])

with col2:
    st.metric("Pages Read", progress["current_page"] - 1)

with col3:
    st.metric("Remaining Pages", remaining_pages)

with col4:
    st.metric("Progress", f"{progress_percentage:.1f}%")

# Progress visualization
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=progress_percentage,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Reading Progress"},
    delta={'reference': 0},
    gauge={
        'axis': {'range': [None, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 25], 'color': "lightgray"},
            {'range': [25, 50], 'color': "yellow"},
            {'range': [50, 75], 'color': "orange"},
            {'range': [75, 100], 'color': "green"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 90
        }
    }
))

fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Reading status
st.header("📈 Reading Status")

col1, col2 = st.columns(2)

with col1:
    if progress["is_finished"]:
        st.success("✅ **Finished Reading**")
    else:
        st.info("📖 **Currently Reading**")

with col2:
    if progress["spoiler_protection"]:
        st.success("🛡️ **Spoiler Protection Enabled**")
    else:
        st.warning("⚠️ **Spoiler Protection Disabled**")

# Query statistics
st.header("💬 Query Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Queries", progress["total_queries"])

with col2:
    last_updated = progress["last_updated"]
    if last_updated:
        last_updated = last_updated[:10]  # Just the date
    st.metric("Last Updated", last_updated or "Never")

with col3:
    # Calculate average queries per page (rough estimate)
    if progress["current_page"] > 0:
        avg_queries = progress["total_queries"] / progress["current_page"]
        st.metric("Avg Queries/Page", f"{avg_queries:.1f}")
    else:
        st.metric("Avg Queries/Page", "0.0")

# System status
st.header("🔧 System Status")

with st.spinner("Checking system status..."):
    try:
        status_response = api_client.get_system_status()
        
        if "error" not in status_response and "data" in status_response:
            system_status = status_response["data"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rag_status = system_status.get("rag_status", {})
                is_ready = rag_status.get("is_ready", False)
                st.metric("RAG System", "Ready" if is_ready else "Not Ready")
            
            with col2:
                vector_storage = system_status.get("vector_storage", {})
                vector_count = vector_storage.get("vector_count", 0)
                st.metric("Vector Count", vector_count)
            
            with col3:
                graph_storage = system_status.get("graph_storage", {})
                graph_available = graph_storage.get("available", False)
                st.metric("Graph Storage", "Available" if graph_available else "Unavailable")
        
        else:
            st.warning("⚠️ Could not retrieve system status")
    
    except Exception as e:
        st.error(f"❌ Error checking system status: {str(e)}")

# Actions
st.header("🎯 Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📖 Update Reading Progress", type="primary"):
        st.switch_page("pages/reading_progress.py")

with col2:
    if st.button("💬 Ask Questions"):
        st.switch_page("pages/query_interface.py")

with col3:
    if st.button("🔄 Reset Document"):
        if st.button("⚠️ Confirm Reset", type="secondary"):
            session_manager.reset_session()
            st.success("✅ Document reset successfully!")
            st.rerun()

# Navigation
st.markdown("---")
st.markdown("**Navigation:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📚 Document Upload"):
        st.switch_page("pages/document_upload.py")

with col2:
    if st.button("📖 Reading Progress"):
        st.switch_page("pages/reading_progress.py")

with col3:
    if st.button("💬 Query Interface"):
        st.switch_page("pages/query_interface.py")

with col4:
    if st.button("🏠 Home"):
        st.switch_page("pages/home.py")
