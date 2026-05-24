# -*- coding: utf-8 -*-
"""
Local filesystem implementation of the StorageBackend interface.
"""
from __future__ import annotations

import os
from pathlib import Path
from ets.core.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """
    LocalStorageBackend saves files to a local root directory.
    Suitable for development, testing, and team-local Docker Compose setups.
    """

    def __init__(self, root_dir: Path | str | None = None):
        if root_dir is None:
            import os
            root_dir = os.getenv("ETS_STORAGE_ROOT", "outputs")
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        # Sanitize path to prevent directory traversal
        sanitized = os.path.normpath(key).lstrip(os.path.sep).lstrip("/")
        return self.root_dir / sanitized

    async def save(self, key: str, data: bytes) -> str:
        dest_path = self._get_path(key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write bytes in a standard blocking manner (can wrap in threadpool if extremely heavy)
        dest_path.write_bytes(data)
        return key

    async def load(self, key: str) -> bytes:
        src_path = self._get_path(key)
        if not src_path.exists():
            raise FileNotFoundError(f"Asset not found in storage: {key}")
        return src_path.read_bytes()

    async def delete(self, key: str) -> None:
        target_path = self._get_path(key)
        if target_path.exists():
            target_path.unlink()

    async def exists(self, key: str) -> bool:
        return self._get_path(key).exists()
