"""Build a fresh, isolated corpus without loading API credentials."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pipeline import build_corpus


if __name__ == "__main__":
    print(build_corpus())
