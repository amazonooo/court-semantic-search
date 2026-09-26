"""Fingerprint of source loaded at process startup; never includes credentials."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_files = sorted((ROOT / "backend" / "app").rglob("*.py")) + [ROOT / "frontend" / "demo.html"]
BUILD_ID = sha256(b"".join(path.read_bytes() for path in _files if path.is_file())).hexdigest()[:12]
