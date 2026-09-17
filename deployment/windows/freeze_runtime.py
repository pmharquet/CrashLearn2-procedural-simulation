"""Writable cache/log locations for the windowed Windows distribution."""

import os
from pathlib import Path
import sys

data = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "CrashLearn"
data.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(data / "cache"))
if sys.stdout is None:
    sys.stdout = (data / "application.log").open("a", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = sys.stdout
