from types import SimpleNamespace

import pytest

from mm_agents.agent import ensure_json_llm_response


def test_json_llm_response_is_accepted():
    response = SimpleNamespace(
        status_code=200,
        headers={"content-type": "application/json; charset=utf-8"},
        text='{"choices": []}',
    )

    ensure_json_llm_response(response, "https://example.test/chat/completions")


def test_waf_html_reports_diagnostics_before_json_parsing():
    response = SimpleNamespace(
        status_code=200,
        headers={
            "content-type": "text/html;charset=UTF-8",
            "server": "Tengine",
            "bxpunish": "1",
            "eagleeye-traceid": "trace-123",
        },
        text=(
            '<a href="https://example.test/punish?'
            'uuid=0123456789abcdef0123456789abcdef&action=deny"></a>'
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        ensure_json_llm_response(
            response,
            "https://example.test/chat/completions",
        )

    message = str(exc_info.value)
    assert "status=200" in message
    assert "content_type=text/html;charset=UTF-8" in message
    assert "server=Tengine" in message
    assert "bxpunish=1" in message
    assert "traceid=trace-123" in message
    assert "waf_uuid=0123456789abcdef0123456789abcdef" in message
    assert "action=deny" not in message
