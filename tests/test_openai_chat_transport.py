import json
from types import SimpleNamespace
from unittest.mock import patch

from mm_agents.transports import OpenAIChatConfig, OpenAIChatTransport


class FakeResponse:
    def __init__(self, *, payload=None, lines=None, content_type="application/json"):
        self.status_code = 200
        self._payload = payload
        self._lines = lines or []
        self.headers = {"content-type": content_type}
        self.text = json.dumps(payload or {})
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        self.closed = True


def test_nonstream_transport_forwards_phoenix_headers_and_qwen_extras():
    response = FakeResponse(
        payload={
            "choices": [{"message": {"content": "DONE", "reasoning_content": "think"}}],
            "usage": {"total_tokens": 3},
        }
    )
    config = OpenAIChatConfig(
        endpoint="http://phoenix/eval/dashscope/chat/completions",
        api_key="tenant",
        emp_id="1",
        iai_tag="tag",
        phoenix_eval_token="token",
        phoenix_domain_proxy="http://iai:7001",
        phoenix_timeout="1200",
    )
    transport = OpenAIChatTransport(config)
    with patch("mm_agents.transports.openai_chat.requests.post", return_value=response) as post:
        result = transport.complete(
            {
                "model": "qwen3.8-max",
                "messages": [],
                "extra_body": {"enable_thinking": True},
                "reasoning_effort": "xhigh",
            }
        )

    assert result.content == "DONE"
    assert result.reasoning_content == "think"
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer tenant"
    assert kwargs["headers"]["x-eval-domain-proxy"] == "http://iai:7001"
    assert kwargs["json"]["enable_thinking"] is True
    assert kwargs["json"]["reasoning_effort"] == "xhigh"
    assert "extra_body" not in kwargs["json"]


def test_stream_transport_keeps_reasoning_and_content():
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"why "}}]}',
        'data: {"choices":[{"delta":{"content":"DONE"}}]}',
        "data: [DONE]",
    ]
    response = FakeResponse(lines=lines, content_type="text/event-stream")
    transport = OpenAIChatTransport(
        OpenAIChatConfig(endpoint="http://sidecar/v1/chat/completions", api_key="key", stream=True)
    )
    with patch("mm_agents.transports.openai_chat.requests.post", return_value=response):
        result = transport.complete({"model": "qwen3.8-max", "messages": []})

    assert result.content == "DONE"
    assert result.reasoning_content == "why "
    assert response.closed


def test_sglang_trajectory_header_is_generic_transport_metadata():
    response = FakeResponse(payload={"choices": [{"message": {"content": "DONE"}}]})
    transport = OpenAIChatTransport(
        OpenAIChatConfig(
            endpoint="https://phoenix/eval/v1/chat/completions",
            api_key="sglang-sidecar",
            phoenix_domain_proxy="http://sglang:30000",
            backend_trajectory_id="approved",
        )
    )
    with patch("mm_agents.transports.openai_chat.requests.post", return_value=response) as post:
        transport.complete({"model": "served-model", "messages": []})

    assert post.call_args.kwargs["headers"]["X-Backend-TrajectoryID"] == "approved"
