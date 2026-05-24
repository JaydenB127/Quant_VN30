# -*- coding: utf-8 -*-
"""
Base feature set abstract class defining the contract for feature engineering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseFeatureSet(ABC):
    """
    Abstract base class for feature engineering pipelines in ETS.
    Provides structural uniformity and anti-leakage guards.
    """

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute the feature matrix.

        Returns
        -------
        pd.DataFrame
            The computed feature DataFrame.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return the names of the constructed features.

        Returns
        -------
        List[str]
            List of feature name strings.
        """
        pass
