"""
Query interface component for the Streamlit frontend.
"""

import streamlit as st
from typing import Dict, Any, Optional
import time
from datetime import datetime

from config import config
from utils.api_client import api_client
from utils.session_manager import session_manager


def render_query_input() -> Optional[str]:
    """Render the query input interface."""
    st.header("💬 Ask Questions About Your Book")
    
    # Check if document is loaded
    if not session_manager.get_current_document():
        st.error("❌ No document loaded. Please upload a document first.")
        if st.button("📚 Go to Document Upload"):
            st.switch_page("pages/document_upload.py")
        return None
    
    # Display current status
    progress = session_manager.get_reading_stats()
    st.info(f"📚 **Current Document:** {session_manager.get_current_document()}")
    st.info(f"📖 **Current Page:** {progress['current_page']} of {progress['total_pages']}")
    
    # Spoiler protection status
    if progress['spoiler_protection']:
        st.success("🛡️ **Spoiler Protection ENABLED** - Queries limited to pages 1-{progress['current_page']}")
    else:
        st.warning("⚠️ **Spoiler Protection DISABLED** - You can ask about any part of the book")
    
    # Query input
    query = st.text_area(
        "Ask a question about your book:",
        placeholder="e.g., Who is the main character? What happens in chapter 3? Explain the relationship between...",
        height=100,
        max_chars=config.MAX_QUERY_LENGTH,
        help=f"Ask questions about characters, plot, themes, or any aspect of the book. Maximum {config.MAX_QUERY_LENGTH} characters."
    )
    
    # Query options
    with st.expander("⚙️ Query Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_method = st.selectbox(
                "Search Method:",
                options=["hybrid", "semantic", "keyword"],
                help="Hybrid combines multiple search methods for best results"
            )
            
            top_k = st.slider(
                "Number of Results:",
                min_value=1,
                max_value=10,
                value=config.DEFAULT_TOP_K,
                help="Number of relevant passages to consider"
            )
        
        with col2:
            similarity_threshold = st.slider(
                "Similarity Threshold:",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum similarity score for results"
            )
            
            include_metadata = st.checkbox(
                "Include Metadata",
                value=True,
                help="Include page numbers and other metadata in results"
            )
    
    # Submit button
    if st.button("🔍 Ask Question", type="primary", disabled=not query.strip()):
        return query.strip()
    
    return None


def render_query_response(response: Dict[str, Any], query: str, processing_time: float):
    """Render the query response."""
    st.header("📝 Answer")
    
    if "error" in response:
        st.error(f"❌ Query failed: {response['error']}")
        return
    
    # Extract response data
    if "data" in response:
        data = response["data"]
    else:
        # Fallback for legacy API response
        data = response
    
    # Display answer
    answer = data.get("answer", "No answer provided")
    st.markdown(f"**Answer:** {answer}")
    
    # Display source pages
    source_pages = data.get("source_pages", [])
    if source_pages:
        st.subheader("📄 Source Pages")
        
        for i, page in enumerate(source_pages, 1):
            with st.expander(f"Source {i}: Page {page.get('page_number', 'Unknown')}"):
                st.write(f"**Content:** {page.get('content', 'No content available')}")
                st.write(f"**Relevance Score:** {page.get('relevance_score', 0):.2f}")
                if page.get('chunk_id'):
                    st.write(f"**Chunk ID:** {page['chunk_id']}")
    
    # Display metadata
    st.subheader("📊 Query Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col2:
        st.metric("Search Method", data.get("search_method", "Unknown"))
    
    with col3:
        st.metric("Confidence Score", f"{data.get('confidence_score', 0):.2f}")
    
    with col4:
        st.metric("Source Pages", len(source_pages))
    
    # Add to query history
    session_manager.add_query_to_history(
        query, 
        answer, 
        session_manager.get_user_progress().get("current_page")
    )


def render_query_history():
    """Render query history."""
    st.header("📚 Query History")
    
    history = session_manager.get_query_history()
    
    if not history:
        st.info("No queries yet. Ask your first question!")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        show_spoiler_protected = st.checkbox("Show spoiler-protected queries", value=True)
    
    with col2:
        if st.button("🗑️ Clear History"):
            session_manager.clear_query_history()
            st.rerun()
    
    # Display history
    for i, query in enumerate(reversed(history)):
        # Filter by spoiler protection
        if not show_spoiler_protected and query.get("spoiler_protected", True):
            continue
        
        with st.expander(f"Query {len(history) - i}: {query['query'][:60]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Question:** {query['query']}")
                st.write(f"**Answer:** {query['response'][:200]}...")
                if query.get("page_number"):
                    st.write(f"**Page:** {query['page_number']}")
            
            with col2:
                st.write(f"**Time:** {query['timestamp'][:19]}")
                if query.get("spoiler_protected", True):
                    st.success("🛡️ Protected")
                else:
                    st.warning("⚠️ Unprotected")


def render_quick_questions():
    """Render quick question suggestions."""
    st.header("💡 Quick Questions")
    st.markdown("Click on a question below to ask it:")
    
    # Get current progress for context-aware suggestions
    progress = session_manager.get_user_progress()
    current_page = progress.get("current_page", 1)
    
    # Context-aware question suggestions
    suggestions = [
        "Who are the main characters?",
        "What is the setting of the story?",
        "What is the main conflict?",
        "What themes are explored?",
        "How does the story begin?",
        "What happens in the first chapter?",
    ]
    
    # Add page-specific suggestions if not finished
    if not progress.get("is_finished", False):
        suggestions.extend([
            f"What happens around page {current_page}?",
            f"Who appears in the first {current_page} pages?",
            f"What is the story about so far?",
        ])
    else:
        suggestions.extend([
            "How does the story end?",
            "What is the resolution?",
            "What are the main themes?",
        ])
    
    # Display suggestions in columns
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                # Set the query in session state to be picked up by the main interface
                st.session_state["quick_query"] = suggestion
                st.rerun()


def render_reading_tips():
    """Render reading tips and help."""
    with st.expander("💡 Reading Tips & Help"):
        st.markdown("""
        **How to Ask Good Questions:**
        
        **Character Questions:**
        - "Who is [character name]?"
        - "What is the relationship between [character A] and [character B]?"
        - "How does [character] change throughout the story?"
        
        **Plot Questions:**
        - "What happens in chapter [X]?"
        - "How does the story begin/end?"
        - "What is the main conflict?"
        
        **Theme Questions:**
        - "What themes are explored in this book?"
        - "How does the author explore [theme]?"
        - "What is the message of the story?"
        
        **Analysis Questions:**
        - "What is the significance of [event/symbol]?"
        - "How does the setting affect the story?"
        - "What literary devices are used?"
        
        **Spoiler Protection:**
        - When enabled, queries are limited to pages up to your current page
        - This prevents spoilers from future chapters
        - Mark the book as finished to disable spoiler protection
        - You can always update your reading progress
        
        **Tips for Better Results:**
        - Be specific in your questions
        - Use character names when possible
        - Ask about specific chapters or pages
        - Try different phrasings if you don't get good results
        """)


def main():
    """Main query interface."""
    st.set_page_config(**config.PAGE_CONFIG)
    
    st.title("💬 Ask Questions About Your Book")
    
    # Check for quick query from suggestions
    if "quick_query" in st.session_state:
        query = st.session_state["quick_query"]
        del st.session_state["quick_query"]
        
        # Process the quick query
        with st.spinner("Processing your question..."):
            start_time = time.time()
            response = api_client.query_document(
                query, 
                session_manager.get_user_progress().get("current_page")
            )
            processing_time = time.time() - start_time
            
            render_query_response(response, query, processing_time)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Ask Question", "Query History", "Quick Questions"])
    
    with tab1:
        query = render_query_input()
        
        if query:
            with st.spinner("Processing your question..."):
                start_time = time.time()
                response = api_client.query_document(
                    query, 
                    session_manager.get_user_progress().get("current_page")
                )
                processing_time = time.time() - start_time
                
                render_query_response(response, query, processing_time)
        
        render_reading_tips()
    
    with tab2:
        render_query_history()
    
    with tab3:
        render_quick_questions()
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Document Upload"):
            st.switch_page("pages/document_upload.py")
    
    with col2:
        if st.button("📖 Reading Progress"):
            st.switch_page("pages/reading_progress.py")
    
    with col3:
        if st.button("📊 Document Info"):
            st.switch_page("pages/document_info.py")


if __name__ == "__main__":
    main()
