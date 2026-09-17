"""Installed, editable and frozen applications share the same resource layout."""

from pathlib import Path
import os

RESOURCES = Path(__file__).resolve().parent / "resources"
MAPS = RESOURCES / "maps"


def default_agent_directory():
    """Use an explicit user setting instead of assuming a neighbouring checkout."""
    configured = os.environ.get("CRASHLEARN_AGENT_DIR")
    return Path(configured).expanduser().resolve() if configured else None
