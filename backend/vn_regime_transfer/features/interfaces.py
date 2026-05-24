# -*- coding: utf-8 -*-
"""
Interface definitions for feature engineering components in the ETS platform.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureSet(ABC):
    """
    Abstract base class for all feature sets.
    Standardizes feature computation and registration.
    """

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute the feature set.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame (typically raw or cleaned OHLCV data).
            
        Returns
        -------
        pd.DataFrame
            The original DataFrame with newly computed feature columns appended.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get names of the features produced by this set.
        
        Returns
        -------
        list of str
            Names of the feature columns.
        """
        pass
