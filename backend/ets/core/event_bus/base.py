# -*- coding: utf-8 -*-
"""
EventBus interface and event classes for the ETS internal event-driven architecture.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class Event:
    """Standard system event model."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "data": self.data})

    @classmethod
    def from_json(cls, json_str: str) -> Event:
        parsed = json.loads(json_str)
        return cls(name=parsed["name"], data=parsed["data"])


class EventBus(ABC):
    """
    Abstract representation of the internal Event Bus.
    Supports publish/subscribe model to loosely couple services.
    """

    @abstractmethod
    async def publish(self, channel: str, event: Event) -> None:
        """Publish an event to a specific channel."""
        pass

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[Event], Any]) -> None:
        """Subscribe to a specific channel and register a callback."""
        pass
