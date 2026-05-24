# -*- coding: utf-8 -*-
"""
Dataset registry managing dataset immutability, versioning, and reproducibility.
Uses SHA-256 content hashing to uniquely identify and deduplicate datasets.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def calculate_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


class DatasetRegistry:
    """
    Registry for tracking dataset metadata and immutability.
    Deduplicates files based on content hash.
    """

    def __init__(self):
        # In-memory registry fallback for Phase 1-2 before DB integration
        self._datasets: Dict[str, Dict[str, Any]] = {}

    def register_dataset(
        self,
        name: str,
        data: bytes,
        file_format: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Generate content hash, verify uniqueness, and register dataset metadata.
        """
        content_hash = calculate_sha256(data)
        
        if content_hash in self._datasets:
            logger.info("Dataset already registered with hash: %s (Deduplicated)", content_hash)
            return self._datasets[content_hash]

        metadata = {
            "name": name,
            "description": description,
            "file_size_bytes": len(data),
            "file_format": file_format.lower(),
            "dataset_hash": content_hash,
            "version": 1,
        }
        
        self._datasets[content_hash] = metadata
        logger.info("Successfully registered dataset '%s' with hash: %s", name, content_hash)
        return metadata

    def get_dataset(self, content_hash: str) -> Optional[Dict[str, Any]]:
        return self._datasets.get(content_hash)
