# -*- coding: utf-8 -*-
"""
Storage abstraction layer for the ETS platform.
Supports local directory and cloud-native object storage backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """
    Abstract base class defining the contract for storage backends.
    Decouples execution pipelines from concrete filesystem / cloud structures.
    """

    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """
        Save raw bytes to the storage backend.

        Parameters
        ----------
        key : str
            The target storage key or path (e.g. "datasets/my_data.csv").
        data : bytes
            The raw data bytes to save.

        Returns
        -------
        str
            The authoritative storage key or URI.
        """
        pass

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """
        Load raw bytes from the storage backend.

        Parameters
        ----------
        key : str
            The storage key or path to retrieve.

        Returns
        -------
        bytes
            The loaded data bytes.
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete the asset from the storage backend.

        Parameters
        ----------
        key : str
            The storage key or path to delete.
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if the asset exists in the storage backend.

        Parameters
        ----------
        key : str
            The storage key or path to check.

        Returns
        -------
        bool
            True if it exists, False otherwise.
        """
        pass
