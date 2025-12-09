"""Compatibility shim that re-exports the production genre_model package."""
from __future__ import annotations

from genre_model import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
