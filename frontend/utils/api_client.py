"""
API client for communicating with the Viggo backend.
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from config import config


class APIClient:
    """Client for communicating with the Viggo API."""
    
    def __init__(self):
        self.base_url = config.api_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Viggo-Frontend/1.0"
        })
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                if files:
                    # Remove Content-Type header for file uploads
                    headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
                    response = requests.post(url, data=data, files=files, headers=headers)
                else:
                    response = self.session.post(url, json=data)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the Viggo API. Please ensure the backend is running.")
            return {"error": "Connection failed"}
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e}")
            return {"error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        return self._make_request("GET", "/health/")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        return self._make_request("GET", "/rag/system")
    
    def upload_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload a document for processing."""
        files = {"file": (filename, file_content, "application/octet-stream")}
        return self._make_request("POST", "/rag/upload", files=files)
    
    def query_document(self, question: str, page_number: Optional[int] = None) -> Dict[str, Any]:
        """Query the document with RAG."""
        data = {
            "question": question,
            "page_number": page_number
        }
        return self._make_request("POST", "/rag/query", data=data)
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the current document."""
        return self._make_request("GET", "/documents/info")
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get supported document formats."""
        return self._make_request("GET", "/documents/supported-formats")
    
    def index_document(self, document_id: str, chunking_strategy: str = "hybrid") -> Dict[str, Any]:
        """Index a document for RAG operations."""
        data = {
            "document_id": document_id,
            "chunking_strategy": chunking_strategy,
            "enable_entity_extraction": True,
            "enable_graph_indexing": True,
            "force_reindex": False
        }
        return self._make_request("POST", "/rag/index", data=data)
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get RAG system status."""
        return self._make_request("GET", "/rag/status")
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self._make_request("GET", "/rag/config")
    
    def update_rag_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update RAG configuration."""
        return self._make_request("PUT", "/rag/config", data=config_data)
    
    def chunk_document(self, document_id: str, chunking_config: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk a document."""
        data = {
            "document_id": document_id,
            **chunking_config
        }
        return self._make_request("POST", "/content/chunk", data=data)
    
    def filter_content(self, content: str, page_number: int) -> Dict[str, Any]:
        """Filter content to determine if it should be indexed."""
        data = {
            "content": content,
            "page_number": page_number,
            "document_type": "book",
            "filter_metadata": True,
            "filter_bibliography": True,
            "filter_preface": False
        }
        return self._make_request("POST", "/content/filter", data=data)
    
    def extract_entities(self, content: str, page_number: int) -> Dict[str, Any]:
        """Extract entities from content."""
        data = {
            "content": content,
            "page_number": page_number,
            "entity_types": ["PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART"],
            "enable_deduplication": True,
            "enable_disambiguation": True,
            "confidence_threshold": 0.7,
            "max_entities": 100
        }
        return self._make_request("POST", "/content/extract-entities", data=data)
    
    def get_chunks(self, page: int = 1, page_size: int = 20, chunk_level: Optional[str] = None) -> Dict[str, Any]:
        """Get document chunks with pagination."""
        params = f"?page={page}&page_size={page_size}"
        if chunk_level:
            params += f"&chunk_level={chunk_level}"
        return self._make_request("GET", f"/content/chunks{params}")


# Global API client instance
api_client = APIClient()
