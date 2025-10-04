"""
Reading progress page for the Viggo Streamlit frontend.
"""

import streamlit as st
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.session_manager import session_manager
from components.reading_progress import main

# Initialize session
session_manager.initialize_session()

# Run the main component
main()
