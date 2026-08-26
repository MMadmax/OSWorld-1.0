"""Provider-neutral transport contracts shared by multimodal agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass
class TransportResponse:
    """Normalized model response without imposing an agent-specific format."""

    content: Any = ""
    reasoning_content: Any = ""
    tool_calls: Any = None
    usage: Any = None
    raw_response: Any = None


@runtime_checkable
class LLMTransport(Protocol):
    """Minimal interface that any mm_agent can consume."""

    def complete(self, payload: Mapping[str, Any]) -> TransportResponse:
        """Execute one model request and return a normalized response."""
