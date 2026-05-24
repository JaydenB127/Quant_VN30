# -*- coding: utf-8 -*-
"""
Redis-backed implementation of the EventBus interface.
Enables real-time synchronization between Celery workers and FastAPI instances.
"""
from __future__ import annotations

import logging
from typing import Any, Callable
from ets.core.event_bus.base import Event, EventBus

logger = logging.getLogger(__name__)


class RedisEventBus(EventBus):
    """
    Redis Pub/Sub async event broker.
    Enables distributed, thread-safe, and process-safe event broadcasts.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                raise ImportError("redis is required. Install with: pip install redis")
        return self._redis

    async def publish(self, channel: str, event: Event) -> None:
        r = self._get_redis()
        # Publish event JSON payload
        await r.publish(channel, event.to_json())

    async def subscribe(self, channel: str, callback: Callable[[Event], Any]) -> None:
        r = self._get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        
        # Start a background loop to listen to messages
        async def listen():
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            event = Event.from_json(message["data"])
                            await callback(event)
                        except Exception as exc:
                            logger.exception("Error processing Redis message on channel '%s': %s", channel, exc)
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.close()

        import asyncio
        asyncio.create_task(listen())
