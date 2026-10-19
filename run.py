"""
AI Video Object Remover - Application Launcher
Launches the FastAPI backend and serves the interactive modern web UI.
"""

import os
import sys
import webbrowser
import uvicorn
from pathlib import Path

# Add project root and modules to python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
AI_ENGINE_DIR = ROOT_DIR / "ai_engine"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    url = f"http://{host}:{port}"

    print("=" * 65)
    print("   AI Video Object Remover (Filmora-style Web Studio)")
    print("=" * 65)
    print(f" * Web Application: {url}")
    print(f" * API Documentation: {url}/docs")
    print(f" * Project Root: {ROOT_DIR}")
    print("=" * 65)
    print("\nStarting server... Press Ctrl+C to stop.\n")

    # Automatically open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Run uvicorn server with hot reload (excluding media/data files)
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=[str(ROOT_DIR), str(BACKEND_DIR), str(ROOT_DIR / "frontend")],
        reload_includes=["*.py", "*.html", "*.css", "*.js"],
        reload_excludes=["*data*", "*uploads*", "*temp*", "*processed*", "*.mp4", "*.jpg", "*.png"],
        log_level="info",
    )

if __name__ == "__main__":
    main()
