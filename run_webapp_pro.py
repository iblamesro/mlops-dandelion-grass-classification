#!/usr/bin/env python3
"""
🌟 Launch Script for Enhanced Streamlit WebApp - PRO VERSION 🌟
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Change to project directory
    os.chdir(project_root)
    
    # Streamlit app path
    app_path = project_root / "src" / "webapp" / "app_pro.py"
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("🌟 Starting Streamlit WebApp - PRO VERSION 🌟")
    print("=" * 60)
    print()
    print("✨ NEW FEATURES:")
    print("  - 🌙 Dark Mode Toggle")
    print("  - 📁 Batch Processing (Multiple Images)")
    print("  - 📊 Prediction History & Statistics")
    print("  - 🎨 Enhanced UI with Animations")
    print("  - 📈 Advanced Visualizations (Gauge, Timeline)")
    print("  - 💾 Export Results (CSV)")
    print("  - 🔍 Image Adjustments (Brightness, Contrast)")
    print()
    print("📱 Access the app at:")
    print("   → http://localhost:8501")
    print()
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    # Run Streamlit
    try:
        subprocess.run([
            "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ WebApp stopped successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
