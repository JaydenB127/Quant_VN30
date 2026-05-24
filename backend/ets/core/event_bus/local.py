# -*- coding: utf-8 -*-
"""
Local (in-memory) implementation of the EventBus interface.
Suitable for testing, local CLI pipeline execution, and single-process contexts.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List
from ets.core.event_bus.base import Event, EventBus

logger = logging.getLogger(__name__)


class LocalEventBus(EventBus):
    """
    In-memory async event broker.
    Dispatches events immediately to registered callback listeners.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = {}

    async def publish(self, channel: str, event: Event) -> None:
        logger.debug("Publishing event '%s' to local channel '%s'", event.name, channel)
        if channel in self._subscribers:
            for callback in self._subscribers[channel]:
                try:
                    # Run callback
                    await callback(event)
                except Exception as exc:
                    logger.exception("Error executing callback on channel '%s' for event '%s': %s",
                                     channel, event.name, exc)

    async def subscribe(self, channel: str, callback: Callable[[Event], Any]) -> None:
        logger.debug("Subscribed callback to local channel '%s'", channel)
        self._subscribers.setdefault(channel, []).append(callback)
