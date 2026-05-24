# -*- coding: utf-8 -*-
"""
Experiment tracking service interface.
Orchestrates parameters logging, sparse/dense metrics logging, and step checkpoints.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
import uuid
from ets.core.event_bus.base import Event, EventBus

logger = logging.getLogger(__name__)


class TrackingService:
    """
    Central tracking service. Handles logging parameters, metrics, step statuses,
    and registering output artifacts. Replaces local text-file logging with
    unified DB and Event-Driven streaming.
    """

    def __init__(self, db_session: Any = None, event_bus: Optional[EventBus] = None):
        self.db_session = db_session
        self.event_bus = event_bus
        self._checkpoints: Dict[str, Dict[str, Any]] = {}  # In-memory backup
        self._run_statuses: Dict[str, str] = {}

    async def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Log a single hyperparameter for a run."""
        logger.info("Run %s | Param: %s = %s", run_id, key, value)
        
        if self.db_session:
            try:
                from db.models import Parameter as DBParameter
                db_param = DBParameter(
                    run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
                    key=key,
                    value=str(value)
                )
                self.db_session.add(db_param)
                await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to log parameter to database: %s", exc)

        # Publish event
        if self.event_bus:
            await self.event_bus.publish(
                f"run:{run_id}",
                Event("parameter.logged", {"run_id": run_id, "key": key, "value": str(value)})
            )

    async def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        """Log a sparse evaluation metric (e.g. final AUC, Sharpe)."""
        logger.debug("Run %s | Metric: %s = %.6f (step %d)", run_id, key, value, step)
        
        if self.db_session:
            try:
                from db.models import Metric as DBMetric
                from datetime import datetime
                db_metric = DBMetric(
                    run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
                    key=key,
                    value=float(value),
                    step=int(step),
                    timestamp=datetime.utcnow()
                )
                self.db_session.add(db_metric)
                await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to log metric to database: %s", exc)

        # Publish event for real-time WebSocket dashboard
        if self.event_bus:
            await self.event_bus.publish(
                f"run:{run_id}",
                Event("metric.logged", {"run_id": run_id, "key": key, "value": value, "step": step})
            )

    async def log_artifact(
        self,
        run_id: str,
        name: str,
        storage_backend: str,
        storage_key: str,
        size_bytes: int,
        content_type: str,
        artifact_type: str = "other",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a computed artifact in the database registry."""
        logger.info("Run %s | Artifact registered: %s (%s, %s)", run_id, name, storage_backend, storage_key)
        
        if self.db_session:
            try:
                from db.models import Artifact as DBArtifact
                db_art = DBArtifact(
                    run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
                    name=name,
                    storage_backend=storage_backend,
                    storage_key=storage_key,
                    file_size_bytes=size_bytes,
                    content_type=content_type,
                    artifact_type=artifact_type,
                    metadata_json=metadata or {}
                )
                self.db_session.add(db_art)
                await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to log artifact to database: %s", exc)

        if self.event_bus:
            await self.event_bus.publish(
                f"run:{run_id}",
                Event("artifact.created", {
                    "run_id": run_id, "name": name,
                    "storage_backend": storage_backend, "storage_key": storage_key,
                    "content_type": content_type, "artifact_type": artifact_type
                })
            )
        return storage_key

    async def log_step_status(
        self,
        run_id: str,
        step_name: str,
        status: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log step progress status for pipeline checkpoint recovery."""
        from datetime import datetime
        run_checkpoints = self._checkpoints.setdefault(run_id, {})
        run_checkpoints[step_name] = {
            "status": status,
            "outputs": outputs or {},
            "error": error,
            "completed_at": time.time() if status == "completed" else None
        }
        logger.info("Run %s | Step Checkpoint: %s -> %s", run_id, step_name, status)
        
        if self.db_session:
            try:
                from db.models import PipelineStep as DBPipelineStep
                from sqlalchemy.future import select
                stmt = select(DBPipelineStep).where(
                    DBPipelineStep.run_id == (uuid.UUID(run_id) if isinstance(run_id, str) else run_id),
                    DBPipelineStep.step_name == step_name
                )
                res = await self.db_session.execute(stmt)
                db_step = res.scalar_one_or_none()
                if db_step:
                    db_step.status = status
                    db_step.outputs = outputs or {}
                    if status == "completed":
                        db_step.completed_at = datetime.utcnow()
                else:
                    db_step = DBPipelineStep(
                        run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
                        step_name=step_name,
                        status=status,
                        outputs=outputs or {},
                        completed_at=datetime.utcnow() if status == "completed" else None
                    )
                    self.db_session.add(db_step)
                await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to log step status to database: %s", exc)

        if self.event_bus:
            await self.event_bus.publish(
                f"run:{run_id}",
                Event("step.completed" if status == "completed" else "step.failed", {
                    "run_id": run_id, "step_name": step_name, "status": status, "error": error
                })
            )

    async def is_step_completed(self, run_id: str, step_name: str) -> bool:
        """Check if a specific pipeline step is already completed for recovery."""
        if self.db_session:
            try:
                from db.models import PipelineStep as DBPipelineStep
                from sqlalchemy.future import select
                stmt = select(DBPipelineStep.status).where(
                    DBPipelineStep.run_id == (uuid.UUID(run_id) if isinstance(run_id, str) else run_id),
                    DBPipelineStep.step_name == step_name
                )
                res = await self.db_session.execute(stmt)
                status = res.scalar_one_or_none()
                return status == "completed"
            except Exception as exc:
                logger.error("Failed to query step status from database: %s", exc)

        run_checkpoints = self._checkpoints.get(run_id, {})
        step_info = run_checkpoints.get(step_name, {})
        return step_info.get("status") == "completed"

    async def get_step_outputs(self, run_id: str, step_name: str) -> Dict[str, Any]:
        """Load outputs of an already-completed step from checkpoints."""
        if self.db_session:
            try:
                from db.models import PipelineStep as DBPipelineStep
                from sqlalchemy.future import select
                stmt = select(DBPipelineStep.outputs).where(
                    DBPipelineStep.run_id == (uuid.UUID(run_id) if isinstance(run_id, str) else run_id),
                    DBPipelineStep.step_name == step_name
                )
                res = await self.db_session.execute(stmt)
                outputs = res.scalar_one_or_none()
                return outputs or {}
            except Exception as exc:
                logger.error("Failed to query step outputs from database: %s", exc)

        run_checkpoints = self._checkpoints.get(run_id, {})
        step_info = run_checkpoints.get(step_name, {})
        return step_info.get("outputs", {})

    async def update_run_status(self, run_id: str, status: str, error_message: Optional[str] = None) -> None:
        """Update overall run lifecycle status."""
        from datetime import datetime
        self._run_statuses[run_id] = status
        logger.info("Run %s | Lifecycle status -> %s", run_id, status)
        
        if self.db_session:
            try:
                from db.models import Run as DBRun
                from sqlalchemy.future import select
                stmt = select(DBRun).where(DBRun.id == (uuid.UUID(run_id) if isinstance(run_id, str) else run_id))
                res = await self.db_session.execute(stmt)
                db_run = res.scalar_one_or_none()
                if db_run:
                    db_run.status = status
                    if status == "running":
                        db_run.started_at = datetime.utcnow()
                    elif status in ("completed", "failed"):
                        db_run.completed_at = datetime.utcnow()
                        if db_run.started_at:
                            db_run.duration_seconds = float((db_run.completed_at - db_run.started_at).total_seconds())
                        if error_message:
                            db_run.error_message = error_message
                    await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to update run status in database: %s", exc)

        if self.event_bus:
            event_name = "run.started" if status == "running" else ("run.completed" if status == "completed" else "run.failed")
            await self.event_bus.publish(
                f"run:{run_id}",
                Event(event_name, {"run_id": run_id, "status": status, "error": error_message})
            )

