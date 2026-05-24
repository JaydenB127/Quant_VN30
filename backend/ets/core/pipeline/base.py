# -*- coding: utf-8 -*-
"""
Pipeline execution engine abstractions.
Implements step-based execution flow, context propagation, and error handling.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


class PipelineContext:
    """
    Mutable context carried across all pipeline execution steps.
    Holds configuration, tracking endpoints, and intermediate step results.
    """

    def __init__(self, config: Dict[str, Any], tracker: Any):
        self.config = config
        self.tracker = tracker
        self.results: Dict[str, Any] = {}
        self.artifacts: Dict[str, bytes] = {}
        self.run_id: Optional[str] = None


class PipelineStep(ABC):
    """
    Abstract representation of a single atomic execution step inside a pipeline.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this step and return the mutated context.
        """
        pass


class PipelineError(Exception):
    """Raised when a pipeline step fails during execution."""
    def __init__(self, step: str, cause: Exception):
        super().__init__(f"Pipeline failed at step '{step}': {cause}")
        self.step = step
        self.cause = cause


class BasePipeline(ABC):
    """
    Abstract parent representing an end-to-end ML workflow pipeline.
    Handles sequential steps orchestration, progress reporting, and recovery.
    """

    def __init__(self, pipeline_type: str):
        self.pipeline_type = pipeline_type

    @abstractmethod
    def get_steps(self) -> List[PipelineStep]:
        """Return the ordered list of execution steps."""
        pass

    async def run(
        self,
        run_id: uuid.UUID | str,
        config: Dict[str, Any],
        tracker: Any,
    ) -> Dict[str, Any]:
        """
        Orchestrate the async execution of pipeline steps sequentially.
        """
        steps = self.get_steps()
        context = PipelineContext(config=config, tracker=tracker)
        run_id_str = str(run_id)
        context.run_id = run_id_str

        logger.info("Starting pipeline '%s' for run: %s", self.pipeline_type, run_id_str)
        
        # In Phase 2, tracking service gets progress updates
        if hasattr(tracker, "log_metric"):
            await tracker.log_metric(run_id_str, "pipeline_progress", 0.0, step=0)

        for i, step in enumerate(steps, 1):
            logger.info("Executing step %d/%d: %s", i, len(steps), step.name)
            
            # Phase 2: Checkpoint recovery hook
            if hasattr(tracker, "is_step_completed") and await tracker.is_step_completed(run_id_str, step.name):
                logger.info("Step '%s' already completed. Restoring outputs from checkpoint.", step.name)
                # Load checkpoint outputs
                outputs = await tracker.get_step_outputs(run_id_str, step.name)
                context.results.update(outputs)
                continue

            try:
                # Mark step as running in checkpoint db if present
                if hasattr(tracker, "log_step_status"):
                    await tracker.log_step_status(run_id_str, step.name, "running")

                context = await step.execute(context)

                # Mark step as completed and save outputs
                if hasattr(tracker, "log_step_status"):
                    await tracker.log_step_status(
                        run_id_str, step.name, "completed", outputs=context.results.get(step.name, {})
                    )
            except Exception as exc:
                logger.exception("Error executing pipeline step '%s': %s", step.name, exc)
                if hasattr(tracker, "log_step_status"):
                    await tracker.log_step_status(run_id_str, step.name, "failed", error=str(exc))
                raise PipelineError(step.name, exc)

            if hasattr(tracker, "log_metric"):
                progress = float(i / len(steps))
                await tracker.log_metric(run_id_str, "pipeline_progress", progress, step=i)

        logger.info("Pipeline '%s' completed successfully for run: %s", self.pipeline_type, run_id_str)
        return context.results
