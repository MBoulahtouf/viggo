"""
Configuration for the Viggo Streamlit frontend.
"""

import os
from typing import Optional

class FrontendConfig:
    """Frontend configuration settings."""
    
    # API Configuration
    API_BASE_URL: str = os.getenv("VIGGO_API_URL", "http://localhost:8000")
    API_VERSION: str = "v1"
    
    # Session State Keys
    SESSION_KEYS = {
        "current_document": "current_document",
        "user_progress": "user_progress",
        "document_info": "document_info",
        "query_history": "query_history",
        "spoiler_protection": "spoiler_protection",
        "is_finished": "is_finished"
    }
    
    # UI Configuration
    PAGE_CONFIG = {
        "page_title": "Viggo - Intelligent Reading Assistant",
        "page_icon": None,
        "layout": "centered",
        "initial_sidebar_state": "expanded"
    }
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = [".pdf", ".epub"]
    
    # Query Configuration
    MAX_QUERY_LENGTH = 1000
    DEFAULT_TOP_K = 5
    
    # Progress Configuration
    MIN_PAGE = 1
    MAX_PAGE = 10000  # Reasonable upper limit
    
    @property
    def api_url(self) -> str:
        """Get the full API base URL."""
        return f"{self.API_BASE_URL}/api/{self.API_VERSION}"
    
    @property
    def health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.api_url}/health"
    
    @property
    def rag_url(self) -> str:
        """Get the RAG operations URL."""
        return f"{self.api_url}/rag"
    
    @property
    def content_url(self) -> str:
        """Get the content processing URL."""
        return f"{self.api_url}/content"
    
    @property
    def documents_url(self) -> str:
        """Get the documents URL."""
        return f"{self.api_url}/documents"

# Global configuration instance
config = FrontendConfig()
