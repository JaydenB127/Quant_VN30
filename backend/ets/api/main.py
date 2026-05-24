# -*- coding: utf-8 -*-
"""
FastAPI application entrypoint.
Registers routers, sets up middlewares, and initializes database connection.
"""
from __future__ import annotations

import sys
import os
# Add backend directory to sys.path so that absolute imports work from both root and backend directory execution contexts
backend_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.session import DatabaseManager
from ets.core.event_bus.local import LocalEventBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("ets.api")

# Global database manager and event bus instance
db_manager = DatabaseManager()
event_bus = LocalEventBus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    logger.info("Initializing database and tables...")
    await db_manager.create_tables()

    # Startup: clean up runs that were stuck 'running' from previous server sessions
    await _cleanup_stuck_runs()
    await _backfill_missing_artifacts()

    yield
    # Shutdown: clean up engines if needed
    logger.info("Shutting down database connections...")
    await db_manager.engine.dispose()


async def _cleanup_stuck_runs():
    """Reset any runs left in 'running' state from previous server sessions."""
    try:
        from db.models import Run as DBRun
        from sqlalchemy.future import select
        from datetime import datetime
        async with db_manager.session_factory() as db:
            stmt = select(DBRun).where(DBRun.status == "running")
            res = await db.execute(stmt)
            stuck = res.scalars().all()
            if stuck:
                logger.warning("Found %d stuck runs from previous session - marking as failed", len(stuck))
                for r in stuck:
                    r.status = "failed"
                    r.error_message = "Server restarted during execution"
                    r.completed_at = datetime.utcnow()
                    if r.started_at:
                        r.duration_seconds = float((r.completed_at - r.started_at).total_seconds())
                await db.commit()
    except Exception as exc:
        logger.error("Failed to clean up stuck runs: %s", exc)


async def _backfill_missing_artifacts():
    """Register artifacts from REPORT_DIR for any completed run missing them.

    Each run gets its own snapshot under REPORT_DIR/runs/<run_id_short>/ so
    subsequent runs don't overwrite each other's files.
    """
    try:
        from db.models import Run as DBRun, Artifact as DBArtifact
        from sqlalchemy.future import select
        from vn_regime_transfer.config import REPORT_DIR

        REPORT_DIR = Path(REPORT_DIR)

        async with db_manager.session_factory() as db:
            stmt = select(DBRun).where(DBRun.status == "completed")
            res = await db.execute(stmt)
            completed_runs = res.scalars().all()

            for run in completed_runs:
                # Skip runs that already have artifacts registered
                art_stmt = select(DBArtifact).where(DBArtifact.run_id == run.id)
                art_res = await db.execute(art_stmt)
                if art_res.scalars().first():
                    continue

                # Create a per-run snapshot directory
                run_id_short = str(run.id)[:8]
                run_report_dir = REPORT_DIR / "runs" / run_id_short
                run_report_dir.mkdir(parents=True, exist_ok=True)

                count = 0
                if REPORT_DIR.is_dir():
                    for filename in sorted(os.listdir(REPORT_DIR)):
                        src = REPORT_DIR / filename
                        if not src.is_file():
                            continue
                        ext = src.suffix.lower()
                        if ext not in (".png", ".csv", ".pdf", ".txt", ".tex"):
                            continue

                        # Copy to per-run snapshot
                        dst = run_report_dir / filename
                        shutil.copy2(str(src), str(dst))

                        content_type = (
                            "image/png" if ext == ".png"
                            else "text/csv" if ext == ".csv"
                            else "application/pdf" if ext == ".pdf"
                            else "text/plain"
                        )
                        artifact_type = (
                            "plot" if ext == ".png"
                            else "report" if ext in (".pdf", ".tex")
                            else "dataset" if ext == ".csv"
                            else "other"
                        )
                        db_art = DBArtifact(
                            run_id=run.id,
                            name=filename,
                            storage_backend="local",
                            storage_key=str(dst),
                            file_size_bytes=dst.stat().st_size,
                            content_type=content_type,
                            artifact_type=artifact_type,
                            metadata_json={"path": str(dst), "run_dir": str(run_report_dir)},
                        )
                        db.add(db_art)
                        count += 1

                if count:
                    await db.commit()
                    logger.info("Backfilled %d artifacts for Run #%s (%s)", count, run.run_number, run_id_short)
    except Exception as exc:
        logger.error("Failed to backfill artifacts: %s", exc)


app = FastAPI(
    title="ETS Core API",
    description="General-purpose Web-based AI Experiment Tracking System Core",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routers
from ets.api.routes import datasets, experiments, runs, views, auth

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(runs.router, prefix="/api/runs", tags=["Runs"])
app.include_router(views.router, prefix="", tags=["Dashboard"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ets-core-api"}


if __name__ == "__main__":
    # Restrict reload directories to prevent writes to ets.db or outputs/ from killing the server mid-run
    reload_dirs = ["ets", "plugins"]
    if os.path.exists("backend"):
        reload_dirs = ["backend/ets", "backend/plugins"]
    uvicorn.run("ets.api.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=reload_dirs)

