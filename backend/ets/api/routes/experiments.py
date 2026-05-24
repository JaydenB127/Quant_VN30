# -*- coding: utf-8 -*-
"""
Experiments router.
Handles experiment CRUD operations and triggering asynchronous pipeline runs.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.session import DatabaseManager
from db.models import Experiment as DBExperiment, Run as DBRun, Dataset as DBDataset
from ets.api.routes.datasets import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class ExperimentCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    dataset_id: uuid.UUID
    pipeline_type: str = "finance_forecasting"
    config_json: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class RunTriggerRequest(BaseModel):
    config_override: Dict[str, Any] = Field(default_factory=dict)

async def execute_run_task(run_id: uuid.UUID, config: Dict[str, Any], pipeline_type: str):
    """
    Background worker task to execute the pipeline run asynchronously.
    Updates run state in DB and streams events via Redis Pub/Sub if active.
    """
    from ets.api.main import db_manager, event_bus
    from ets.core.tracking.buffer import BufferedTrackingService

    # 1. Fetch appropriate pipeline
    if pipeline_type == "finance_forecasting":
        from plugins.finance.pipeline import FinanceForecastingPipeline
        pipeline = FinanceForecastingPipeline()
    else:
        logger.error("Unknown pipeline type: %s", pipeline_type)
        return

    # 2. Setup separate session for background thread
    async with db_manager.session_factory() as session:
        # Create performance-optimized buffered tracking service with event bus for real-time telemetry
        tracker = BufferedTrackingService(db_session=session, event_bus=event_bus, buffer_size=10, flush_interval_seconds=1.0)

        try:
            logger.info("Background Run %s | Initializing execution...", run_id)
            await tracker.update_run_status(str(run_id), "running")

            # Log all initial parameters
            for k, v in config.items():
                if isinstance(v, (dict, list)):
                    await tracker.log_parameter(str(run_id), k, str(v))
                else:
                    await tracker.log_parameter(str(run_id), k, v)

            # Run the modular pipeline asynchronously
            results = await pipeline.run(
                run_id=run_id,
                config=config,
                tracker=tracker,
            )

            logger.info("Background Run %s | Completed successfully!", run_id)
            await tracker.update_run_status(str(run_id), "completed")

        except Exception as exc:
            logger.exception("Background Run %s | Execution failed: %s", run_id, exc)
            await tracker.update_run_status(str(run_id), "failed", error_message=str(exc))
        finally:
            # Clean up and flush tracker buffer
            await tracker.close()


@router.post("/")
async def create_experiment(
    exp: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new experiment associated with a dataset."""
    # Verify dataset exists
    stmt = select(DBDataset).where(DBDataset.id == exp.dataset_id)
    res = await db.execute(stmt)
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db_exp = DBExperiment(
        name=exp.name,
        description=exp.description,
        dataset_id=exp.dataset_id,
        pipeline_type=exp.pipeline_type,
        config_json=exp.config_json,
        tags=exp.tags,
    )
    db.add(db_exp)
    await db.commit()

    return {
        "id": str(db_exp.id),
        "name": db_exp.name,
        "dataset_id": str(db_exp.dataset_id),
        "pipeline_type": db_exp.pipeline_type,
        "created_at": db_exp.created_at.isoformat(),
    }


@router.get("/")
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """List all experiments."""
    stmt = select(DBExperiment).order_by(DBExperiment.created_at.desc())
    res = await db.execute(stmt)
    experiments = res.scalars().all()
    return [
        {
            "id": str(e.id),
            "name": e.name,
            "description": e.description,
            "dataset_id": str(e.dataset_id) if e.dataset_id else None,
            "pipeline_type": e.pipeline_type,
            "config_json": e.config_json,
            "tags": e.tags,
            "created_at": e.created_at.isoformat(),
        }
        for e in experiments
    ]


@router.post("/{experiment_id}/run")
async def trigger_experiment_run(
    experiment_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    request: Optional[RunTriggerRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a new asynchronous execution run for an experiment.
    Returns immediately with run ID, executing the pipeline in the background.
    """
    # 1. Verify experiment exists
    stmt = select(DBExperiment).where(DBExperiment.id == experiment_id)
    res = await db.execute(stmt)
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # 2. Resolve the linked dataset to get the actual CSV path
    dataset_csv_path = None
    dataset_name = "VN30 (default)"
    if exp.dataset_id:
        stmt_ds = select(DBDataset).where(DBDataset.id == exp.dataset_id)
        res_ds = await db.execute(stmt_ds)
        dataset = res_ds.scalar_one_or_none()
        if dataset:
            dataset_name = dataset.name
            base_dir = os.path.abspath(".")
            candidate = os.path.join(base_dir, dataset.storage_key)
            candidate_outputs = os.path.join(base_dir, "outputs", dataset.storage_key)
            if os.path.exists(candidate):
                dataset_csv_path = candidate
                logger.info("Run will use dataset CSV: %s", candidate)
            elif os.path.exists(candidate_outputs):
                dataset_csv_path = candidate_outputs
                logger.info("Run will use dataset CSV (from outputs): %s", candidate_outputs)
            else:
                logger.warning("Dataset file not found at %s or %s — falling back to VN30 download", candidate, candidate_outputs)

    # 3. Compute run number sequentially
    stmt_num = select(DBRun.run_number).where(DBRun.experiment_id == experiment_id).order_by(DBRun.run_number.desc()).limit(1)
    res_num = await db.execute(stmt_num)
    last_num = res_num.scalar_one_or_none()
    run_number = (last_num or 0) + 1

    # 4. Create run record in database
    db_run = DBRun(
        experiment_id=experiment_id,
        run_number=run_number,
        status="created",
        config_snapshot=exp.config_json,
    )
    db.add(db_run)
    await db.commit()

    # 5. Build pipeline config with run identity and dataset path
    pipeline_config = exp.config_json.copy()
    if request and request.config_override:
        pipeline_config.update(request.config_override)

    pipeline_config.setdefault("skip_download", True)
    pipeline_config.setdefault("fast", True)
    pipeline_config.setdefault("quick", True)
    pipeline_config.setdefault("run_dl", False)

    # Inject run-specific metadata so pipeline stores artifacts per-run
    pipeline_config["run_id"] = str(db_run.id)
    pipeline_config["experiment_name"] = exp.name
    pipeline_config["dataset_name"] = dataset_name
    if dataset_csv_path:
        pipeline_config["dataset_csv_path"] = dataset_csv_path

    background_tasks.add_task(
        execute_run_task,
        run_id=db_run.id,
        config=pipeline_config,
        pipeline_type=exp.pipeline_type,
    )

    logger.info("Run %s triggered for experiment '%s' (Run #%d) with dataset '%s'", db_run.id, exp.name, run_number, dataset_name)
    return {
        "run_id": str(db_run.id),
        "run_number": run_number,
        "status": "created",
    }


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete an experiment, cascade-deleting its runs, database records, and disk files."""
    # 1. Fetch the experiment
    stmt = select(DBExperiment).where(DBExperiment.id == experiment_id)
    res = await db.execute(stmt)
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # 2. Fetch all runs associated with this experiment
    stmt_runs = select(DBRun).where(DBRun.experiment_id == experiment_id)
    res_runs = await db.execute(stmt_runs)
    runs = res_runs.scalars().all()

    # Reuse run deletion logic to cleanly remove each run's files from disk
    from ets.api.routes.runs import delete_run
    for run in runs:
        try:
            await delete_run(run_id=run.id, db=db)
        except Exception as e:
            logger.warning("Failed to cascade delete run %s: %s", run.id, e)

    # 3. Delete experiment itself from DB (will cascade delete remaining records)
    await db.delete(exp)
    await db.commit()
    logger.info("Successfully deleted experiment %s (%s)", exp.name, experiment_id)
    return {"status": "success", "message": f"Experiment '{exp.name}' and all its runs deleted successfully"}

@router.put("/{experiment_id}")
async def update_experiment(
    experiment_id: uuid.UUID,
    exp_update: ExperimentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an existing experiment."""
    stmt = select(DBExperiment).where(DBExperiment.id == experiment_id)
    res = await db.execute(stmt)
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if exp_update.name is not None:
        exp.name = exp_update.name
    if exp_update.description is not None:
        exp.description = exp_update.description
    if exp_update.tags is not None:
        exp.tags = exp_update.tags

    await db.commit()
    return {
        "id": str(exp.id),
        "name": exp.name,
        "description": exp.description,
        "dataset_id": str(exp.dataset_id) if exp.dataset_id else None,
        "pipeline_type": exp.pipeline_type,
        "config_json": exp.config_json,
        "tags": exp.tags,
        "created_at": exp.created_at.isoformat(),
    }


