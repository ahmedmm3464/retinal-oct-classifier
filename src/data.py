"""Data loading, cleaning, splitting and preprocessing for the OCT project."""

import hashlib
from pathlib import Path

import pandas as pd


def file_hash(path):
    """Return the MD5 fingerprint of a file's bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def index_files(root):
    """Return a table with the relative path, name and hash of every file."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root!r}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "relpath": path.relative_to(root).as_posix(),
                "name": path.name,
                "md5": file_hash(path),
            })
    return pd.DataFrame(rows)
