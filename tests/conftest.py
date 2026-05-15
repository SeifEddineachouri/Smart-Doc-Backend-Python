"""Test configuration for making the application package importable.

Pytest sometimes runs with a working directory that does not automatically
place the repository root on ``sys.path`` in CI. This keeps imports like
``from app.main import app`` working reliably.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

