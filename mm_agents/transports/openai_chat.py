"""OpenAI Chat Completions transport for Phoenix, sidecars, and direct APIs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

import requests

from .base import TransportResponse


logger = logging.getLogger("desktopenv.transport.openai_chat")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def ensure_json_llm_response(response, endpoint: str) -> None:
    """Fail with useful Phoenix/WAF diagnostics before JSON parsing."""
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return

    body_prefix = response.text[:4096]
    waf_uuid_match = re.search(
        r"(?:[?&]|\b)uuid=([0-9a-fA-F]{16,64})",
        body_prefix,
    )
    waf_uuid = waf_uuid_match.group(1) if waf_uuid_match else None
    raise RuntimeError(
        "Non-JSON response from OpenAI-compatible endpoint: "
        f"endpoint={endpoint}, status={response.status_code}, "
        f"content_type={content_type or None}, "
        f"server={response.headers.get('server')}, "
        f"bxpunish={response.headers.get('bxpunish')}, "
        f"traceid={response.headers.get('eagleeye-traceid')}, "
        f"waf_uuid={waf_uuid}"
    )


def _message_response(data: Mapping[str, Any], *, raw_response=None) -> TransportResponse:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response contains no choices")
    message = choices[0].get("message") or {}
    return TransportResponse(
        content=message.get("content", ""),
        reasoning_content=message.get("reasoning_content", ""),
        tool_calls=message.get("tool_calls"),
        usage=data.get("usage"),
        raw_response=raw_response if raw_response is not None else data,
    )


def consume_openai_stream(response, endpoint: str) -> TransportResponse:
    """Consume OpenAI-compatible SSE while retaining content and reasoning."""
    content_type = response.headers.get("content-type", "").lower()
    logger.info("Opened LLM stream: content_type=%s", content_type or None)
    if "application/json" in content_type:
        try:
            return _message_response(response.json(), raw_response=response)
        finally:
            response.close()
    if "text/event-stream" not in content_type:
        try:
            ensure_json_llm_response(response, endpoint)
        finally:
            response.close()

    content_parts = []
    reasoning_parts = []
    tool_calls = []
    events = []
    usage = None
    saw_event = False
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            line = raw_line.strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            events.append(event)
            if not saw_event:
                logger.info("Received first LLM stream event")
            saw_event = True
            if event.get("error"):
                raise RuntimeError(
                    f"Streaming error from OpenAI-compatible endpoint: {event['error']}"
                )
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or choice.get("message") or {}
                content_parts.append(_content_text(delta.get("content")))
                reasoning_parts.append(_content_text(delta.get("reasoning_content")))
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
    finally:
        response.close()

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and not reasoning and not tool_calls:
        raise RuntimeError(
            "OpenAI-compatible stream ended without content: "
            f"endpoint={endpoint}, saw_event={saw_event}"
        )
    return TransportResponse(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls or None,
        usage=usage,
        raw_response=events,
    )


@dataclass(frozen=True)
class OpenAIChatConfig:
    endpoint: str
    api_key: str
    request_timeout: float = 180.0
    stream: bool = False
    emp_id: str | None = None
    iai_tag: str | None = None
    phoenix_eval_token: str | None = None
    phoenix_domain_proxy: str | None = None
    phoenix_timeout: str | None = None
    backend_trajectory_id: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAIChatConfig":
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        endpoint = os.environ.get("OSWORLD_OPENAI_CHAT_COMPLETIONS_URL")
        if not endpoint:
            endpoint = (
                f"{base_url.rstrip('/')}/chat/completions"
                if base_url.rstrip("/").endswith("/v1")
                else f"{base_url.rstrip('/')}/v1/chat/completions"
            )
        return cls(
            endpoint=endpoint,
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            request_timeout=float(os.environ.get("OSWORLD_LLM_REQUEST_TIMEOUT", "180")),
            stream=_env_bool("OSWORLD_LLM_STREAM", False),
            emp_id=os.environ.get("OSWORLD_EMP_ID"),
            iai_tag=os.environ.get("OSWORLD_IAI_TAG"),
            phoenix_eval_token=os.environ.get("PHOENIX_EVAL_TOKEN"),
            phoenix_domain_proxy=os.environ.get("PHOENIX_DOMAIN_PROXY"),
            phoenix_timeout=os.environ.get("PHOENIX_EVAL_TIMEOUT"),
            backend_trajectory_id=(
                os.environ.get("OSWORLD_BACKEND_TRAJECTORY_ID")
                or os.environ.get("SANDBOX_TRAJECTORY_ID")
            ),
        )


class OpenAIChatTransport:
    """Requests-based transport preserving the repository's Phoenix behavior."""

    def __init__(self, config: OpenAIChatConfig | None = None):
        self.config = config or OpenAIChatConfig.from_env()

    def _headers(self) -> dict[str, str]:
        config = self.config
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        optional = {
            "empId": config.emp_id,
            "iai-tag": config.iai_tag,
            "x-eval-token": config.phoenix_eval_token,
            "x-eval-domain-proxy": config.phoenix_domain_proxy,
            "x-eval-timeout": config.phoenix_timeout,
            "X-Backend-TrajectoryID": config.backend_trajectory_id,
        }
        headers.update({key: value for key, value in optional.items() if value})
        if config.stream:
            headers["Accept"] = "text/event-stream"
        return headers

    @staticmethod
    def _wire_payload(payload: Mapping[str, Any], stream: bool) -> dict[str, Any]:
        request_payload = dict(payload)
        extra_body = request_payload.pop("extra_body", None)
        if isinstance(extra_body, Mapping):
            for key, value in extra_body.items():
                request_payload.setdefault(key, value)
        if stream:
            request_payload["stream"] = True
        return request_payload

    def _post(self, payload: Mapping[str, Any]):
        return requests.post(
            self.config.endpoint,
            headers=self._headers(),
            json=self._wire_payload(payload, self.config.stream),
            timeout=self.config.request_timeout,
            stream=self.config.stream,
        )

    def _read_success(self, response) -> TransportResponse:
        if self.config.stream:
            return consume_openai_stream(response, self.config.endpoint)
        ensure_json_llm_response(response, self.config.endpoint)
        return _message_response(response.json(), raw_response=response)

    def complete(self, payload: Mapping[str, Any]) -> TransportResponse:
        response = self._post(payload)
        if response.status_code == 200:
            return self._read_success(response)

        ensure_json_llm_response(response, self.config.endpoint)
        error = response.json().get("error") or {}
        if error.get("code") == "context_length_exceeded":
            logger.error("Context length exceeded. Retrying with a smaller context.")
            shortened = dict(payload)
            messages = list(shortened.get("messages") or [])
            if len(messages) > 2:
                shortened["messages"] = [messages[0], messages[-1]]
            retry_response = self._post(shortened)
            if retry_response.status_code == 200:
                return self._read_success(retry_response)
            ensure_json_llm_response(retry_response, self.config.endpoint)
            logger.error("Failed after shortening model history: %s", retry_response.text)
            return TransportResponse(raw_response=retry_response)

        logger.error("Failed to call LLM: %s", response.text)
        time.sleep(5)
        return TransportResponse(raw_response=response)
