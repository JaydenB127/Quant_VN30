# -*- coding: utf-8 -*-
"""
Buffered tracking service subclass.
Caches high-frequency dense telemetry metrics in-memory and flushes in bulk batches.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from ets.core.event_bus.base import Event, EventBus
from ets.core.tracking.service import TrackingService

logger = logging.getLogger(__name__)


class BufferedTrackingService(TrackingService):
    """
    Performance-optimized metrics logger.
    Buffers high-frequency batch/epoch metrics to avoid PostgreSQL write amplification.
    """

    def __init__(
        self,
        db_session: Any = None,
        event_bus: Optional[EventBus] = None,
        buffer_size: int = 100,
        flush_interval_seconds: float = 2.0,
    ):
        super().__init__(db_session=db_session, event_bus=event_bus)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        
        # Buffer: run_id -> List of metrics
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._active = True

        # Start background periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    async def log_dense_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        """
        Log a high-frequency metric (e.g. training loss per batch, epoch accuracy).
        Buffers in memory instead of writing directly to the database.
        """
        async with self._lock:
            run_buf = self._buffer.setdefault(run_id, [])
            run_buf.append({
                "key": key,
                "value": float(value),
                "step": int(step),
                "timestamp": float(asyncio.get_event_loop().time())
            })
            
            # Flush immediately if buffer size exceeded
            if len(run_buf) >= self.buffer_size:
                await self._flush_run_buffer(run_id)

    async def _flush_run_buffer(self, run_id: str) -> None:
        """Bulk inserts or dispatches the buffered metrics."""
        metrics_to_flush = self._buffer.get(run_id, [])
        if not metrics_to_flush:
            return
            
        self._buffer[run_id] = []
        logger.info("Flushing %d buffered metrics for run %s", len(metrics_to_flush), run_id)

        # 1. Real-time Pub/Sub bulk dispatch (optional grouping)
        if self.event_bus:
            # Publish grouped metrics telemetry event
            await self.event_bus.publish(
                f"run:{run_id}",
                Event("metrics.flushed", {"run_id": run_id, "metrics": metrics_to_flush})
            )

        # 2. Database bulk write hook
        if self.db_session:
            try:
                from db.models import Metric as DBMetric
                import uuid
                from datetime import datetime
                db_metrics = [
                    DBMetric(
                        run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
                        key=m["key"],
                        value=m["value"],
                        step=m["step"],
                        timestamp=datetime.utcnow()
                    )
                    for m in metrics_to_flush
                ]
                self.db_session.add_all(db_metrics)
                await self.db_session.commit()
            except Exception as exc:
                logger.error("Failed to bulk write metrics to database: %s", exc)

    async def flush_all(self) -> None:
        """Manually flush all buffers immediately (e.g. at pipeline completion)."""
        async with self._lock:
            for run_id in list(self._buffer.keys()):
                await self._flush_run_buffer(run_id)

    async def close(self) -> None:
        """Close tracking service, flush pending buffers, and terminate worker thread."""
        self._active = False
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush_all()

    async def _periodic_flush_loop(self) -> None:
        """Background loop to periodically flush metrics buffer."""
        while self._active:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self._lock:
                    for run_id in list(self._buffer.keys()):
                        await self._flush_run_buffer(run_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in metrics flush loop: %s", exc)
