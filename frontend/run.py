#!/usr/bin/env python3
"""
Startup script for the Viggo Streamlit frontend.

This script provides a convenient way to start the Streamlit application
with proper configuration and error handling.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import requests
        import pandas
        import plotly
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_backend_connection():
    """Check if the backend is accessible."""
    try:
        import requests
        from config import config
        
        response = requests.get(f"{config.api_url}/health/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is accessible")
            return True
        else:
            print(f"⚠️ Backend returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Cannot connect to backend: {e}")
        print("Make sure the Viggo backend is running on the configured URL")
        return False

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Start the Viggo Streamlit frontend")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and backend checks")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting Viggo Frontend...")
    print(f"📱 Streamlit will be available at: http://{args.host}:{args.port}")
    
    # Check dependencies
    if not args.skip_checks:
        print("\n🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("\n🔍 Checking backend connection...")
        check_backend_connection()
    
    # Prepare Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
        print("🐛 Running in debug mode")
    
    # Set environment variables
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    print(f"\n🎯 Starting Streamlit with command: {' '.join(cmd)}")
    print("📚 Viggo Frontend is starting up...")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Start Streamlit
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Viggo Frontend...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
