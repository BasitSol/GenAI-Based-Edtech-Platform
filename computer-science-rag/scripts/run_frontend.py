"""Run the plain React/Vite frontend without coupling it to Python UI code."""
from __future__ import annotations

import shutil
import subprocess

from backend.shared.core import ROOT


if __name__ == "__main__":
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise RuntimeError("Node.js and npm are required to run the React frontend.")
    subprocess.run([npm, "run", "dev"], cwd=ROOT / "frontend", check=True)
