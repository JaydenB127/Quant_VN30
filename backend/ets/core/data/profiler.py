# -*- coding: utf-8 -*-
"""
Generic Dataset Profiler for schema inference and data analysis.
Automatically infers data types, calculates column statistics,
and suggests ML problem types (classification, regression, forecasting).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    NLP = "nlp"
    UNKNOWN = "unknown"


class DatasetProfiler:
    """
    Analyzes tabular datasets, infers schemas, and recommends pipelines.
    Domain-agnostic and robust to dirty / incomplete data.
    """

    @staticmethod
    def _to_native(value: Any) -> Any:
        """Convert numpy/pandas scalar types to JSON-serializable Python native types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: DatasetProfiler._to_native(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DatasetProfiler._to_native(v) for v in value]
        return value

    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Infer schema types and calculate statistics for every column.
        """
        schema = {}
        for col in df.columns:
            series = df[col]
            inferred_type = self._detect_column_type(series)
            
            # Compute stats
            stats = self._compute_column_stats(series, inferred_type)
            
            schema[col] = {
                "type": inferred_type.value,
                "statistics": stats,
            }
        return schema

    def suggest_problem_type(self, df: pd.DataFrame, target_col: str, schema: Dict[str, Dict[str, Any]]) -> ProblemType:
        """
        Auto-detect the best ML problem type based on target column characteristics.
        """
        if target_col not in df.columns:
            return ProblemType.UNKNOWN

        target_series = df[target_col].dropna()
        target_info = schema[target_col]
        target_type = target_info["type"]

        # Text column suggestions -> NLP
        if target_type == ColumnType.TEXT:
            return ProblemType.NLP

        # Categorical or boolean target -> Classification
        if target_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            n_unique = target_series.nunique()
            if n_unique == 2:
                return ProblemType.BINARY_CLASSIFICATION
            return ProblemType.MULTICLASS_CLASSIFICATION

        # Numeric targets -> Regression or Forecasting
        if target_type == ColumnType.NUMERIC:
            # Check for potential time-series/forecasting pattern
            has_datetime = any(info["type"] == ColumnType.DATETIME.value for info in schema.values())
            if has_datetime:
                # If dataframe index or a column is sorted datetime, highly likely forecasting
                return ProblemType.FORECASTING
            return ProblemType.REGRESSION

        return ProblemType.UNKNOWN

    def suggest_target(self, schema: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Suggest a likely target column (e.g. columns named 'label', 'target', 'direction')."""
        target_keywords = ["label", "target", "direction", "class", "y", "close_change", "fwd_return"]
        for col_name in schema.keys():
            if any(kw in col_name.lower() for kw in target_keywords):
                return col_name
        return None

    def _detect_column_type(self, series: pd.Series) -> ColumnType:
        """Rule-based column type inference chain."""
        # 1. Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        try:
            # Try to convert object/string to datetime if it looks like date
            if series.dtype == object:
                sample = series.dropna().head(10)
                if len(sample) > 0 and all(isinstance(val, str) and len(val) >= 8 for val in sample):
                    pd.to_datetime(sample, errors="raise")
                    return ColumnType.DATETIME
        except (ValueError, TypeError):
            pass

        # 2. Check if boolean
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN
        if series.nunique() == 2:
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0, "0", "1", "True", "False", "true", "false"}):
                return ColumnType.BOOLEAN

        # 3. Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERIC

        # 4. Check if text or category
        if series.dtype == object:
            # High unique count relative to row count suggests free text
            non_null = series.dropna()
            if len(non_null) > 0:
                avg_tokens = non_null.astype(str).str.split().str.len().mean()
                if avg_tokens > 3.0:
                    return ColumnType.TEXT
            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _compute_column_stats(self, series: pd.Series, col_type: ColumnType) -> Dict[str, Any]:
        """Compute standard statistics based on inferred column type."""
        n_rows = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique())

        stats = {
            "total_count": n_rows,
            "missing_count": n_missing,
            "missing_ratio": float(n_missing / n_rows) if n_rows > 0 else 0.0,
            "unique_count": n_unique,
        }

        if col_type == ColumnType.NUMERIC:
            stats.update({
                "mean": float(series.mean()) if pd.notna(series.mean()) else None,
                "std": float(series.std()) if pd.notna(series.std()) else None,
                "min": float(series.min()) if pd.notna(series.min()) else None,
                "max": float(series.max()) if pd.notna(series.max()) else None,
                "median": float(series.median()) if pd.notna(series.median()) else None,
            })
        elif col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            mode = series.mode()
            stats.update({
                "most_frequent": mode.iloc[0] if not mode.empty else None,
                "most_frequent_frequency": int(series.value_counts().iloc[0]) if n_unique > 0 else 0,
            })
        elif col_type == ColumnType.DATETIME:
            dt_series = pd.to_datetime(series, errors="coerce").dropna()
            stats.update({
                "min_date": dt_series.min().strftime("%Y-%m-%d %H:%M:%S") if not dt_series.empty else None,
                "max_date": dt_series.max().strftime("%Y-%m-%d %H:%M:%S") if not dt_series.empty else None,
            })

        return self._to_native(stats)
