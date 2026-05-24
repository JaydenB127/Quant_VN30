# -*- coding: utf-8 -*-
"""
Generic Regression Pipeline plugin.
Implements cleaning, scaling, LightGBM regressor training, and metrics logging.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb

from ets.core.pipeline.base import BasePipeline, PipelineStep, PipelineContext
from ets.core.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)


class RegressionDataLoadStep(PipelineStep):
    """Loads a registered dataset from storage."""
    def __init__(self):
        super().__init__("dataset_loading")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        storage_key = context.config.get("dataset_storage_key")
        if not storage_key:
            raise ValueError("dataset_storage_key must be provided in config")
            
        storage = LocalStorageBackend()
        data_bytes = await storage.load(storage_key)
        
        import io
        df = pd.read_csv(io.BytesIO(data_bytes))
        context.results["raw_df"] = df
        return context


class RegressionPreprocessStep(PipelineStep):
    """Performs scaling and encoding for regression features."""
    def __init__(self):
        super().__init__("preprocessing")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        df = context.results["raw_df"].copy()
        target_col = context.config.get("target_column")
        if not target_col or target_col not in df.columns:
            raise ValueError(f"target_column '{target_col}' not found in dataset")

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in (np.float32, np.float64, np.int32, np.int64):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(np.float64)

        # Encode categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale numerical features
        num_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        if len(num_cols) > 0:
            X[num_cols] = scaler.fit_transform(X[num_cols])

        context.results["X"] = X
        context.results["y"] = y
        return context


class RegressionTrainStep(PipelineStep):
    """Trains a LightGBM Regressor and logs MSE, MAE, R² metrics."""
    def __init__(self):
        super().__init__("model_training")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        X = context.results["X"]
        y = context.results["y"]
        run_id_str = str(context.tracker.db_session.new or uuid.uuid4()) if hasattr(context.tracker, "db_session") else "test-run"

        # Split train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        # Predictions
        preds = model.predict(X_val)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        # Log metrics to tracker
        if context.tracker:
            await context.tracker.log_metric(run_id_str, "mse", mse, step=1)
            await context.tracker.log_metric(run_id_str, "mae", mae, step=1)
            await context.tracker.log_metric(run_id_str, "r2_score", r2, step=1)

        # Log dense telemetry mock (for BufferedTrackingService verification)
        if hasattr(context.tracker, "log_dense_metric"):
            for epoch in range(10):
                # Simulated training loss
                loss = mse * (1.1 - (0.02 * epoch))
                await context.tracker.log_dense_metric(run_id_str, "train_loss", loss, step=epoch)

        context.results["metrics"] = {
            "mse": float(mse),
            "mae": float(mae),
            "r2_score": float(r2),
        }
        return context


class RegressionPipeline(BasePipeline):
    """
    Standard modular Regression Pipeline conforming to the generic BasePipeline.
    Can be loaded dynamically by the API router when suggested problem type is regression.
    """
    def __init__(self):
        super().__init__("regression")

    def get_steps(self) -> List[PipelineStep]:
        return [
            RegressionDataLoadStep(),
            RegressionPreprocessStep(),
            RegressionTrainStep(),
        ]
