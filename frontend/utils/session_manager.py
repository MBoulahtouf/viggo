"""
Session state management for the Streamlit frontend.
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime

from config import config


class SessionManager:
    """Manages session state for the Streamlit app."""
    
    def __init__(self):
        self.keys = config.SESSION_KEYS
    
    def initialize_session(self):
        """Initialize session state with default values."""
        if self.keys["current_document"] not in st.session_state:
            st.session_state[self.keys["current_document"]] = None
        
        if self.keys["user_progress"] not in st.session_state:
            st.session_state[self.keys["user_progress"]] = {
                "current_page": 0,  # Start at 0 (not started)
                "is_finished": False,
                "last_updated": datetime.now().isoformat(),
                "total_pages": 0,
                "spoiler_protection": True  # Enable spoiler protection by default
            }
        
        if self.keys["document_info"] not in st.session_state:
            st.session_state[self.keys["document_info"]] = None
        
        if self.keys["query_history"] not in st.session_state:
            st.session_state[self.keys["query_history"]] = []
        
        if self.keys["spoiler_protection"] not in st.session_state:
            st.session_state[self.keys["spoiler_protection"]] = True
        
        if self.keys["is_finished"] not in st.session_state:
            st.session_state[self.keys["is_finished"]] = False
    
    def get_current_document(self) -> Optional[str]:
        """Get the current document filename."""
        return st.session_state.get(self.keys["current_document"])
    
    def set_current_document(self, filename: str):
        """Set the current document filename."""
        st.session_state[self.keys["current_document"]] = filename
    
    def get_user_progress(self) -> Dict[str, Any]:
        """Get user reading progress."""
        return st.session_state.get(self.keys["user_progress"], {})
    
    def update_user_progress(self, current_page: int, is_finished: bool = False):
        """Update user reading progress."""
        progress = self.get_user_progress()
        progress.update({
            "current_page": current_page,
            "is_finished": is_finished,
            "last_updated": datetime.now().isoformat()
        })
        st.session_state[self.keys["user_progress"]] = progress
        
        # Update spoiler protection based on finished status
        st.session_state[self.keys["spoiler_protection"]] = not is_finished
        st.session_state[self.keys["is_finished"]] = is_finished
    
    def get_document_info(self) -> Optional[Dict[str, Any]]:
        """Get document information."""
        return st.session_state.get(self.keys["document_info"])
    
    def set_document_info(self, info: Dict[str, Any]):
        """Set document information."""
        st.session_state[self.keys["document_info"]] = info
        
        # Update total pages in progress
        if info and "total_pages" in info:
            progress = self.get_user_progress()
            progress["total_pages"] = info["total_pages"]
            st.session_state[self.keys["user_progress"]] = progress
    
    def add_query_to_history(self, query: str, response: str, page_number: Optional[int] = None):
        """Add a query to the history."""
        history = st.session_state.get(self.keys["query_history"], [])
        history.append({
            "query": query,
            "response": response,
            "page_number": page_number,
            "timestamp": datetime.now().isoformat(),
            "spoiler_protected": st.session_state.get(self.keys["spoiler_protection"], True)
        })
        st.session_state[self.keys["query_history"]] = history
    
    def get_query_history(self) -> list:
        """Get query history."""
        return st.session_state.get(self.keys["query_history"], [])
    
    def clear_query_history(self):
        """Clear query history."""
        st.session_state[self.keys["query_history"]] = []
    
    def is_spoiler_protection_enabled(self) -> bool:
        """Check if spoiler protection is enabled."""
        progress = self.get_user_progress()
        return progress.get("spoiler_protection", True)
    
    def is_finished(self) -> bool:
        """Check if the user has finished the book."""
        return st.session_state.get(self.keys["is_finished"], False)
    
    def reset_session(self):
        """Reset all session state."""
        for key in self.keys.values():
            if key in st.session_state:
                del st.session_state[key]
        self.initialize_session()
    
    def get_progress_percentage(self) -> float:
        """Get reading progress as a percentage."""
        progress = self.get_user_progress()
        current_page = progress.get("current_page", 1)
        total_pages = progress.get("total_pages", 1)
        
        if total_pages <= 0:
            return 0.0
        
        return min(100.0, (current_page / total_pages) * 100.0)
    
    def get_reading_stats(self) -> Dict[str, Any]:
        """Get reading statistics."""
        progress = self.get_user_progress()
        history = self.get_query_history()
        
        return {
            "current_page": progress.get("current_page", 1),
            "total_pages": progress.get("total_pages", 0),
            "progress_percentage": self.get_progress_percentage(),
            "is_finished": self.is_finished(),
            "total_queries": len(history),
            "last_updated": progress.get("last_updated"),
            "spoiler_protection": self.is_spoiler_protection_enabled()
        }
    
    def update_reading_progress(self, current_page: int):
        """Update reading progress."""
        self.update_user_progress(current_page)
        st.rerun()
    
    def mark_as_finished(self):
        """Mark the book as finished."""
        progress = self.get_user_progress()
        total_pages = progress.get("total_pages", 0)
        if total_pages > 0:
            self.update_user_progress(total_pages, is_finished=True)
        else:
            # If we don't know total pages, just mark as finished
            progress.update({
                "is_finished": True,
                "last_updated": datetime.now().isoformat()
            })
            st.session_state[self.keys["user_progress"]] = progress
        st.rerun()
    
    def toggle_spoiler_protection(self):
        """Toggle spoiler protection on/off."""
        progress = self.get_user_progress()
        current_setting = progress.get("spoiler_protection", True)
        progress.update({
            "spoiler_protection": not current_setting,
            "last_updated": datetime.now().isoformat()
        })
        st.session_state[self.keys["user_progress"]] = progress


# Global session manager instance
session_manager = SessionManager()
