# strider/core/state.py

# This module holds the in-memory state of the application,
# replacing the need for global variables in the main file.

# Information about the loaded document
document_metadata = {}

# Raw text extracted from the document, filtered for content
document_store = []

# Text chunks and their source page for RAG
all_chunks_with_metadata = []

# The searchable FAISS vector index
vector_index = None
