"""Explicit Streamlit entry point that loads local runtime configuration."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ROOT, load_runtime_environment


if __name__ == "__main__":
    load_runtime_environment()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app" / "app.py")], check=True)
