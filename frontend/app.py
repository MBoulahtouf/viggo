"""
Main Streamlit application for Viggo - Intelligent Reading Assistant.

This is the entry point for the Streamlit frontend that provides a user-friendly
interface for the Viggo RAG system with spoiler protection and reading progress tracking.
"""

import streamlit as st
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from utils.session_manager import session_manager

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Initialize session state
session_manager.initialize_session()

# Main app title
st.title("📚 Viggo - Intelligent Reading Assistant")
st.markdown("""
Welcome to **Viggo**, your intelligent reading companion that helps you explore and understand your books 
through AI-powered question answering with spoiler protection.
""")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")

# Check API connection
with st.sidebar:
    with st.spinner("Checking API..."):
        try:
            from utils.api_client import api_client
            health_response = api_client.health_check()
            if "error" not in health_response:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except Exception as e:
            st.error("❌ Connection Failed")

# Navigation menu
st.sidebar.markdown("### 📚 Document Management")
if st.sidebar.button("📤 Upload Document", use_container_width=True):
    st.switch_page("pages/document_upload.py")

if st.sidebar.button("📊 Document Info", use_container_width=True):
    st.switch_page("pages/document_info.py")

st.sidebar.markdown("### 📖 Reading")
if st.sidebar.button("📈 Reading Progress", use_container_width=True):
    st.switch_page("pages/reading_progress.py")

if st.sidebar.button("💬 Ask Questions", use_container_width=True):
    st.switch_page("pages/query_interface.py")

st.sidebar.markdown("### 🏠")
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.switch_page("pages/home.py")

# Current status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")

current_doc = session_manager.get_current_document()
if current_doc:
    st.sidebar.success(f"📚 {current_doc}")
    
    progress = session_manager.get_reading_stats()
    st.sidebar.metric("Progress", f"{progress['progress_percentage']:.1f}%")
    
    if progress["is_finished"]:
        st.sidebar.success("✅ Finished")
    else:
        st.sidebar.info(f"📖 Page {progress['current_page']}")
    
    if progress["spoiler_protection"]:
        st.sidebar.success("🛡️ Protected")
    else:
        st.sidebar.warning("⚠️ Unprotected")
else:
    st.sidebar.info("📝 No document loaded")

# Main content area
st.markdown("## 🚀 Welcome to Viggo!")

# Check if document is loaded
if current_doc:
    st.success(f"📚 **Current Document:** {current_doc}")
    
    # Quick stats
    progress = session_manager.get_reading_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Page", progress["current_page"])
    
    with col2:
        st.metric("Total Pages", progress["total_pages"])
    
    with col3:
        st.metric("Progress", f"{progress['progress_percentage']:.1f}%")
    
    with col4:
        st.metric("Queries", progress["total_queries"])
    
    # Quick actions
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Ask Questions", type="primary", use_container_width=True):
            st.switch_page("pages/query_interface.py")
    
    with col2:
        if st.button("📖 Update Progress", use_container_width=True):
            st.switch_page("pages/reading_progress.py")
    
    with col3:
        if st.button("📊 View Details", use_container_width=True):
            st.switch_page("pages/document_info.py")

else:
    st.info("📝 **No document loaded** - Upload a document to get started!")
    
    # Get started section
    st.markdown("### 🚀 Get Started")
    
    st.markdown("""
    **To start using Viggo:**
    
    1. **📤 Upload a Document** - Upload a PDF or EPUB file
    2. **📈 Set Reading Progress** - Tell us what page you're on
    3. **💬 Ask Questions** - Get AI-powered answers with spoiler protection
    4. **📊 Track Progress** - Keep your reading progress up to date
    
    **Key Features:**
    - 🛡️ **Spoiler Protection** - Queries limited to pages you've read
    - 🤖 **AI-Powered** - Advanced RAG system for accurate answers
    - 📊 **Progress Tracking** - Visualize your reading journey
    - 💬 **Smart Queries** - Ask about characters, plot, themes, and more
    """)
    
    if st.button("📚 Upload Your First Document", type="primary", use_container_width=True):
        st.switch_page("pages/document_upload.py")

# Features overview
st.markdown("### ✨ Features")

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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Viggo - Intelligent Reading Assistant | Built with Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)
