"""
Home page for the Viggo Streamlit frontend.
"""

import streamlit as st
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.session_manager import session_manager
from utils.api_client import api_client

# Initialize session
session_manager.initialize_session()

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Main title and description
st.title("📚 Viggo - Intelligent Reading Assistant")
st.markdown("""
Welcome to **Viggo**, your intelligent reading companion that helps you explore and understand your books 
through AI-powered question answering with spoiler protection.
""")

# Check API connection
with st.spinner("Checking system status..."):
    try:
        health_response = api_client.health_check()
        if "error" not in health_response:
            st.success("✅ Connected to Viggo API")
        else:
            st.error("❌ Cannot connect to Viggo API. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")

# Current status
st.header("📊 Current Status")

current_doc = session_manager.get_current_document()
progress = session_manager.get_reading_stats()

if current_doc:
    st.success(f"📚 **Current Document:** {current_doc}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Page", progress["current_page"])
    
    with col2:
        st.metric("Total Pages", progress["total_pages"])
    
    with col3:
        st.metric("Progress", f"{progress['progress_percentage']:.1f}%")
    
    # Reading status
    if progress["is_finished"]:
        st.success("✅ **Finished Reading** - Spoiler protection disabled")
    else:
        st.info(f"📖 **Currently Reading** - Spoiler protection enabled (up to page {progress['current_page']})")
    
    # Quick actions
    st.header("🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Ask Questions", type="primary"):
            st.switch_page("pages/query_interface.py")
    
    with col2:
        if st.button("📖 Update Progress"):
            st.switch_page("pages/reading_progress.py")
    
    with col3:
        if st.button("📊 Document Info"):
            st.switch_page("pages/document_info.py")

else:
    st.info("📝 **No document loaded** - Upload a document to get started!")
    
    # Get started section
    st.header("🚀 Get Started")
    
    st.markdown("""
    **To start using Viggo:**
    
    1. **Upload a Document** - Upload a PDF or EPUB file
    2. **Set Reading Progress** - Tell us what page you're on
    3. **Ask Questions** - Get AI-powered answers with spoiler protection
    4. **Update Progress** - Keep your reading progress up to date
    
    **Features:**
    - 🛡️ **Spoiler Protection** - Queries limited to pages you've read
    - 🤖 **AI-Powered** - Advanced RAG system for accurate answers
    - 📊 **Progress Tracking** - Visualize your reading journey
    - 💬 **Smart Queries** - Ask about characters, plot, themes, and more
    """)
    
    if st.button("📚 Upload Document", type="primary", use_container_width=True):
        st.switch_page("pages/document_upload.py")

# Features overview
st.header("✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🛡️ Spoiler Protection**
    - Automatically limits queries to pages you've read
    - Prevents accidental spoilers from future chapters
    - Disable when you finish the book
    
    **🤖 Intelligent Answers**
    - AI-powered question answering
    - Context-aware responses
    - Source page references
    """)

with col2:
    st.markdown("""
    **📊 Progress Tracking**
    - Visual reading progress
    - Reading statistics
    - Query history
    
    **💬 Smart Queries**
    - Ask about characters, plot, themes
    - Quick question suggestions
    - Context-aware recommendations
    """)

# Navigation
st.header("🧭 Navigation")

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
    if st.button("📊 Document Info"):
        st.switch_page("pages/document_info.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Viggo - Intelligent Reading Assistant | Built with Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)
