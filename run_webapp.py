#!/usr/bin/env python3
"""
Launch script for the Streamlit WebApp
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import subprocess
    
    webapp_path = project_root / "src" / "webapp" / "app_enhanced.py"
    
    print("🌼 Starting Streamlit WebApp...")
    print(f"📂 App location: {webapp_path}")
    print("🌐 Local URL: http://localhost:8501")
    print("📚 Features:")
    print("   - Upload images for classification")
    print("   - Real-time predictions")
    print("   - Confidence scores visualization")
    print("   - Local model or API backend")
    print("\n🚀 Launching...")
    print("-" * 60)
    
    subprocess.run([
        "streamlit", "run",
        str(webapp_path),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.serverAddress", "localhost"
    ])
