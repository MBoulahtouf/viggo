"""
Reading progress component for the Streamlit frontend.
"""

import streamlit as st
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config import config
from utils.session_manager import session_manager


def render_progress_setup():
    """
    Render the reading progress setup interface.
    
    Returns:
        Tuple of (current_page, is_finished) if setup is complete
    """
    st.header("📖 Reading Progress Setup")
    st.markdown("Tell us about your reading progress to enable spoiler protection.")
    
    document_info = session_manager.get_document_info()
    if not document_info:
        st.error("❌ No document loaded. Please upload a document first.")
        return None, None
    
    # Get document info
    doc_info = document_info.get("document_info", {})
    total_pages = doc_info.get("total_pages", 100)  # Default fallback
    
    st.info(f"📚 **Document:** {session_manager.get_current_document()}")
    st.info(f"📄 **Total Pages:** {total_pages}")
    
    # Current page input
    st.subheader("📍 Current Page")
    current_page = st.number_input(
        "What page are you currently on?",
        min_value=1,
        max_value=total_pages,
        value=1,
        help=f"Enter the page number you're currently reading (1-{total_pages})"
    )
    
    # Finished status
    st.subheader("🏁 Reading Status")
    is_finished = st.radio(
        "Have you finished this book?",
        options=[False, True],
        format_func=lambda x: "Yes, I've finished reading" if x else "No, I'm still reading",
        help="Mark as finished to disable spoiler protection"
    )
    
    # Progress visualization
    progress_percentage = (current_page / total_pages) * 100 if total_pages > 0 else 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Progress bar
        st.progress(progress_percentage / 100)
        st.caption(f"Progress: {current_page}/{total_pages} pages ({progress_percentage:.1f}%)")
    
    with col2:
        # Status indicator
        if is_finished:
            st.success("✅ Finished")
        else:
            st.info("📖 Reading")
    
    # Spoiler protection info
    st.subheader("🛡️ Spoiler Protection")
    if is_finished:
        st.warning("⚠️ **Spoiler protection is DISABLED** - You can ask about any part of the book")
    else:
        st.success("✅ **Spoiler protection is ENABLED** - Queries will be limited to pages up to your current page")
        st.caption(f"Queries will only return information from pages 1-{current_page}")
    
    # Save progress button
    if st.button("💾 Save Reading Progress", type="primary"):
        session_manager.update_user_progress(current_page, is_finished)
        st.success("✅ Reading progress saved!")
        st.rerun()
    
    return current_page, is_finished


def render_progress_update():
    """Render the reading progress update interface."""
    st.header("📈 Update Reading Progress")
    
    progress = session_manager.get_user_progress()
    current_page = progress.get("current_page", 1)
    is_finished = progress.get("is_finished", False)
    total_pages = progress.get("total_pages", 100)
    
    st.info(f"📚 **Current Progress:** Page {current_page} of {total_pages}")
    
    # Quick page update
    st.subheader("📍 Quick Page Update")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Previous Page"):
            new_page = max(1, current_page - 1)
            session_manager.update_user_progress(new_page, is_finished)
            st.rerun()
    
    with col2:
        new_page = st.number_input(
            "Go to page:",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            key="page_input"
        )
        if st.button("📖 Go to Page"):
            session_manager.update_user_progress(new_page, is_finished)
            st.rerun()
    
    with col3:
        if st.button("➡️ Next Page"):
            new_page = min(total_pages, current_page + 1)
            session_manager.update_user_progress(new_page, is_finished)
            st.rerun()
    
    # Finished status toggle
    st.subheader("🏁 Reading Status")
    new_finished_status = st.radio(
        "Reading status:",
        options=[False, True],
        index=1 if is_finished else 0,
        format_func=lambda x: "Finished reading" if x else "Still reading",
        key="finished_status"
    )
    
    if new_finished_status != is_finished:
        if st.button("💾 Update Status"):
            session_manager.update_user_progress(current_page, new_finished_status)
            st.success("✅ Reading status updated!")
            st.rerun()


def render_progress_visualization():
    """Render reading progress visualization."""
    st.header("📊 Reading Progress Visualization")
    
    progress = session_manager.get_user_progress()
    current_page = progress.get("current_page", 1)
    total_pages = progress.get("total_pages", 100)
    is_finished = progress.get("is_finished", False)
    
    if total_pages <= 0:
        st.warning("No document loaded or invalid page count.")
        return
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Page", current_page)
    
    with col2:
        st.metric("Total Pages", total_pages)
    
    with col3:
        progress_percentage = (current_page / total_pages) * 100
        st.metric("Progress", f"{progress_percentage:.1f}%")
    
    with col4:
        remaining_pages = max(0, total_pages - current_page)
        st.metric("Remaining", remaining_pages)
    
    # Progress chart
    fig = go.Figure()
    
    # Add progress bar
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=progress_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Reading Progress"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
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
    
    # Reading timeline
    st.subheader("📅 Reading Timeline")
    
    # Mock reading data (in a real app, this would come from actual reading sessions)
    reading_sessions = [
        {"date": "2024-01-01", "pages_read": 10, "start_page": 1, "end_page": 10},
        {"date": "2024-01-02", "pages_read": 15, "start_page": 11, "end_page": 25},
        {"date": "2024-01-03", "pages_read": 20, "start_page": 26, "end_page": 45},
        {"date": "2024-01-04", "pages_read": 5, "start_page": 46, "end_page": 50},
    ]
    
    if reading_sessions:
        df = st.session_state.get("reading_data", None)
        if df is None:
            import pandas as pd
            df = pd.DataFrame(reading_sessions)
            df['date'] = pd.to_datetime(df['date'])
            st.session_state["reading_data"] = df
        
        # Pages read over time
        fig = px.bar(
            df, 
            x='date', 
            y='pages_read',
            title="Pages Read Over Time",
            labels={'pages_read': 'Pages Read', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative progress
        df['cumulative_pages'] = df['pages_read'].cumsum()
        fig = px.line(
            df, 
            x='date', 
            y='cumulative_pages',
            title="Cumulative Reading Progress",
            labels={'cumulative_pages': 'Total Pages Read', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_reading_stats():
    """Render reading statistics."""
    st.header("📈 Reading Statistics")
    
    stats = session_manager.get_reading_stats()
    history = session_manager.get_query_history()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Progress Stats")
        st.metric("Current Page", stats["current_page"])
        st.metric("Total Pages", stats["total_pages"])
        st.metric("Progress", f"{stats['progress_percentage']:.1f}%")
        st.metric("Finished", "Yes" if stats["is_finished"] else "No")
    
    with col2:
        st.subheader("💬 Query Stats")
        st.metric("Total Queries", stats["total_queries"])
        st.metric("Spoiler Protection", "Enabled" if stats["spoiler_protection"] else "Disabled")
        st.metric("Last Updated", stats["last_updated"][:10] if stats["last_updated"] else "Never")
    
    # Query history summary
    if history:
        st.subheader("📝 Recent Queries")
        for i, query in enumerate(history[-5:]):  # Show last 5 queries
            with st.expander(f"Query {len(history) - i}: {query['query'][:50]}..."):
                st.write(f"**Question:** {query['query']}")
                st.write(f"**Page:** {query['page_number'] or 'Any'}")
                st.write(f"**Time:** {query['timestamp'][:19]}")
                st.write(f"**Spoiler Protected:** {'Yes' if query['spoiler_protected'] else 'No'}")


def main():
    """Main reading progress interface."""
    st.set_page_config(**config.PAGE_CONFIG)
    
    st.title("📖 Reading Progress")
    
    # Check if document is loaded
    if not session_manager.get_current_document():
        st.error("❌ No document loaded. Please upload a document first.")
        if st.button("📚 Go to Document Upload"):
            st.switch_page("pages/document_upload.py")
        return
    
    # Tab interface
    tab1, tab2, tab3, tab4 = st.tabs(["Setup", "Update", "Visualization", "Statistics"])
    
    with tab1:
        render_progress_setup()
    
    with tab2:
        render_progress_update()
    
    with tab3:
        render_progress_visualization()
    
    with tab4:
        render_reading_stats()
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Document Upload"):
            st.switch_page("pages/document_upload.py")
    
    with col2:
        if st.button("💬 Start Querying"):
            st.switch_page("pages/query_interface.py")
    
    with col3:
        if st.button("📊 Document Info"):
            st.switch_page("pages/document_info.py")


if __name__ == "__main__":
    main()
