# -*- coding: utf-8 -*-
"""
Datasets router.
Handles file ingestion, immutable registration, SHA-256 hashing, and automated profiling.
"""
from __future__ import annotations

import logging
import uuid
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.session import DatabaseManager
from db.models import Dataset as DBDataset
from ets.core.data.registry import DatasetRegistry
from ets.core.data.profiler import DatasetProfiler
from ets.core.storage.local import LocalStorageBackend

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get db session from global db_manager
async def get_db() -> AsyncSession:
    from ets.api.main import db_manager
    async for session in db_manager.get_session():
        yield session


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest and register a new dataset.
    Performs immutable SHA-256 validation, profiles the schema, and saves to storage.
    """
    logger.info("Ingesting uploaded file: %s", file.filename)
    
    # 1. Read raw bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2. Immutable hash verification
    registry = DatasetRegistry()
    metadata = registry.register_dataset(
        name=file.filename or "unknown",
        data=contents,
        file_format=file.filename.split(".")[-1] if file.filename else "csv",
        description=description,
    )
    dataset_hash = metadata["dataset_hash"]

    # 3. Check for existing identical dataset to enforce deduplication
    stmt = select(DBDataset).where(DBDataset.dataset_hash == dataset_hash)
    res = await db.execute(stmt)
    existing_db_dataset = res.scalar_one_or_none()
    if existing_db_dataset:
        logger.info("Dataset deduplicated! Returning existing record for hash: %s", dataset_hash)
        return {
            "id": str(existing_db_dataset.id),
            "name": existing_db_dataset.name,
            "dataset_hash": existing_db_dataset.dataset_hash,
            "status": "deduplicated",
            "suggested_target": existing_db_dataset.suggested_target,
            "suggested_problem_type": existing_db_dataset.suggested_problem_type,
        }

    # 4. Save to Local Storage Backend
    storage = LocalStorageBackend()
    storage_key = f"datasets/{dataset_hash}.csv"
    await storage.save(storage_key, contents)

    # 5. Parse and profile data
    try:
        import io
        # Parse CSV
        df = pd.read_csv(io.BytesIO(contents))
        n_rows = len(df)
        n_columns = len(df.columns)
        
        # Run generic DatasetProfiler
        profiler = DatasetProfiler()
        schema_json = profiler.infer_schema(df)
        suggested_target = profiler.suggest_target(schema_json)
        
        suggested_problem_type = "unknown"
        if suggested_target:
            suggested_problem_type = profiler.suggest_problem_type(df, suggested_target, schema_json).value
        
        profile_json = {
            "columns": list(df.columns),
            "n_missing": int(df.isna().sum().sum()),
            "n_rows": int(n_rows),
            "n_columns": int(n_columns),
        }
    except Exception as exc:
        logger.error("Failed to parse and profile dataset: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}")

    # 6. Save metadata to Database
    db_dataset = DBDataset(
        name=file.filename or "Unnamed Dataset",
        description=description,
        storage_backend="local",
        storage_key=storage_key,
        file_size_bytes=len(contents),
        file_format=metadata["file_format"],
        n_rows=n_rows,
        n_columns=n_columns,
        dataset_hash=dataset_hash,
        schema_json=schema_json,
        profile_json=profile_json,
        suggested_target=suggested_target,
        suggested_problem_type=suggested_problem_type,
        version=1,
    )
    
    db.add(db_dataset)
    await db.commit()
    logger.info("Successfully registered dataset in DB with ID: %s", db_dataset.id)

    return {
        "id": str(db_dataset.id),
        "name": db_dataset.name,
        "dataset_hash": db_dataset.dataset_hash,
        "status": "created",
        "n_rows": n_rows,
        "n_columns": n_columns,
        "suggested_target": suggested_target,
        "suggested_problem_type": suggested_problem_type,
    }


@router.get("/")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all registered datasets."""
    stmt = select(DBDataset).order_by(DBDataset.created_at.desc())
    res = await db.execute(stmt)
    datasets = res.scalars().all()
    return [
        {
            "id": str(d.id),
            "name": d.name,
            "description": d.description,
            "file_size_bytes": d.file_size_bytes,
            "file_format": d.file_format,
            "n_rows": d.n_rows,
            "n_columns": d.n_columns,
            "dataset_hash": d.dataset_hash,
            "created_at": d.created_at.isoformat(),
        }
        for d in datasets
    ]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Get details, schema and profile for a specific dataset."""
    stmt = select(DBDataset).where(DBDataset.id == dataset_id)
    res = await db.execute(stmt)
    d = res.scalar_one_or_none()
    if not d:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    return {
        "id": str(d.id),
        "name": d.name,
        "description": d.description,
        "storage_backend": d.storage_backend,
        "storage_key": d.storage_key,
        "file_size_bytes": d.file_size_bytes,
        "file_format": d.file_format,
        "n_rows": d.n_rows,
        "n_columns": d.n_columns,
        "dataset_hash": d.dataset_hash,
        "schema_json": d.schema_json,
        "profile_json": d.profile_json,
        "suggested_target": d.suggested_target,
        "suggested_problem_type": d.suggested_problem_type,
        "created_at": d.created_at.isoformat(),
    }


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset, its database record, and its file from disk if not referenced by experiments."""
    from db.models import Experiment
    
    # 1. Fetch the dataset
    stmt = select(DBDataset).where(DBDataset.id == dataset_id)
    res = await db.execute(stmt)
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 2. Check if dataset is referenced by experiments
    stmt_exp = select(Experiment).where(Experiment.dataset_id == dataset_id)
    res_exp = await db.execute(stmt_exp)
    if res_exp.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="Cannot delete dataset because it is referenced by existing experiments. Delete the experiments first."
        )

    # 3. Delete file from disk
    import os
    base_dir = os.path.abspath(".")
    file_path = os.path.join(base_dir, "outputs", dataset.storage_key)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info("Deleted dataset file: %s", file_path)
        except Exception as e:
            logger.warning("Failed to delete dataset file %s: %s", file_path, e)

    # 4. Delete DB record
    await db.delete(dataset)
    await db.commit()
    logger.info("Successfully deleted dataset %s (%s)", dataset.name, dataset_id)
    return {"status": "success", "message": f"Dataset '{dataset.name}' deleted successfully"}
