"""Make the team's existing modules importable from the orchestrator (docproc, pipeline, backend)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

for sub in ("backend", "pipeline", "docproc"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
