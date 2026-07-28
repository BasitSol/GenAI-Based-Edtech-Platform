"""Explicit API entry point that loads local runtime configuration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from backend.shared.core import load_runtime_environment


if __name__ == "__main__":
    load_runtime_environment()
    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000, reload=True)
