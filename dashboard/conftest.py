"""Collection guard for the dashboard's doctests.

The dashboard needs the optional ``dashboard`` extra.  Skip the whole
folder rather than failing collection when the library is installed on
its own.
"""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("streamlit") is None:
    collect_ignore_glob = ["*.py"]
