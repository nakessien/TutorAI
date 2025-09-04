import os
import sys
import yaml
import asyncio
import subprocess
import threading
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def setup_directories(config):
    """Create necessary directories"""
    paths = config.get("paths", {})
    dirs_to_create = [
        paths.get("data_dir", "./data"),
        paths.get("documents_dir", "./data/documents"),
        paths.get("vectors_dir", "./data/vectors"),
        paths.get("logs_dir", "./logs"),
        paths.get("models_dir", "./data/models"),
        paths.get("temp_dir", "./temp"),
    ]

    for directory in dirs_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """Check if required dependencies are installed"""
    # 修正的依赖检查 - 使用正确的导入名称
    dependencies = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("sentence_transformers", "sentence-transformers"),
        ("fitz", "PyMuPDF"),  # 关键修正：导入名是fitz
        ("faiss", "faiss-cpu"),
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("pydantic", "pydantic")
    ]

    missing_packages = []
    for import_name, package_name in dependencies:
        try:
            __import__(import_name.replace("-", "_"))
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name}")

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False

    return True


async def initialize_services(config):
    """Initialize core services"""
    print("Initializing services...")

    try:
        # 使用更宽松的导入检查
        try:
            from core.llm_service import LLMService
            print("✓ LLM service available")
        except ImportError as e:
            print(f"⚠ LLM service not available: {e}")

        try:
            from core.rag_service import RAGService
            print("✓ RAG service available")
        except ImportError as e:
            print(f"⚠ RAG service not available: {e}")

        try:
            from core.preference_db import PreferenceDatabase
            print("✓ Preference database available")
        except ImportError as e:
            print(f"⚠ Preference database not available: {e}")

        print("✓ Services initialization completed")
        return True

    except Exception as e:
        print(f"✗ Service initialization failed: {e}")
        return False


def start_backend(config):
    """Start FastAPI backend server"""
    try:
        from api.backend import app
        import uvicorn

        api_config = config.get("api", {})
        host = api_config.get("host", "127.0.0.1")
        port = api_config.get("port", 8000)

        print(f"Starting backend server at http://{host}:{port}")
        print(f"API docs available at: http://{host}:{port}/docs")

        uvicorn.run(
            "api.backend:app",
            host=host,
            port=port,
            reload=api_config.get("reload", False),
            log_level="info"
        )
    except ImportError as e:
        print(f"Failed to start backend: {e}")
        print("Make sure all dependencies are installed.")


def start_frontend(config):
    """Start Streamlit frontend"""
    frontend_config = config.get("frontend", {})
    port = 8501

    print(f"Starting frontend at http://localhost:{port}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "ui/frontend.py",
        "--server.port", str(port),
        "--server.address", "localhost"
    ]

    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Failed to start frontend: {e}")


def main():
    """Main entry point"""
    print("🎓 TutorAI - Intelligent Academic Advisor")
    print("=" * 50)

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Using defaults.")
        config = {}

    # Setup directories
    setup_directories(config)
    print("✓ Directories created")

    # Check dependencies with better error handling
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n⚠ Some dependencies are missing, but continuing anyway...")
        print("Some features may not work properly.")
    else:
        print("✓ All dependencies verified")

    # Initialize services
    services_ok = asyncio.run(initialize_services(config))
    if not services_ok:
        print("\n⚠ Service initialization had issues, but continuing...")

    # Start services based on command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "backend":
        start_backend(config)
    elif mode == "frontend":
        start_frontend(config)
    elif mode == "full":
        print("\nStarting full system...")
        # Start backend in a separate thread
        backend_thread = threading.Thread(
            target=start_backend,
            args=(config,),
            daemon=True
        )
        backend_thread.start()

        # Wait a moment for backend to start
        import time
        print("Waiting for backend to initialize...")
        time.sleep(3)

        # Start frontend in main thread
        start_frontend(config)
    else:
        print("Usage: python main.py [backend|frontend|full]")
        print("  backend  - Start only the API server")
        print("  frontend - Start only the Streamlit UI")
        print("  full     - Start both (default)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 TutorAI shutdown complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)