import os
import unittest
from unittest.mock import patch

from mm_agents.agent import PromptAgent


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "DONE"}}]}


class PhoenixSGLangRoutingTest(unittest.TestCase):
    def test_backend_trajectory_id_is_forwarded_to_phoenix(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            return FakeResponse()

        env = {
            "OSWORLD_OPENAI_COMPATIBLE": "1",
            "OPENAI_API_KEY": "sglang-sidecar",
            "OSWORLD_OPENAI_CHAT_COMPLETIONS_URL": (
                "https://phoenix.example/eval/v1/chat/completions"
            ),
            "PHOENIX_EVAL_TOKEN": "test-token",
            "PHOENIX_DOMAIN_PROXY": "http://sglang.example:30000",
            "PHOENIX_EVAL_TIMEOUT": "300",
            "OSWORLD_BACKEND_TRAJECTORY_ID": "approved-trajectory",
            "OSWORLD_LLM_STREAM": "0",
        }
        with patch.dict(os.environ, env), patch(
            "mm_agents.agent.requests.post", side_effect=fake_post
        ):
            agent = PromptAgent(model="served-model", action_space="pyautogui")
            response = agent.call_llm({"model": "served-model", "messages": []})

        self.assertEqual(response, "DONE")
        self.assertTrue(captured["url"].endswith("/eval/v1/chat/completions"))
        self.assertEqual(
            captured["headers"]["x-eval-domain-proxy"],
            "http://sglang.example:30000",
        )
        self.assertEqual(
            captured["headers"]["X-Backend-TrajectoryID"],
            "approved-trajectory",
        )


if __name__ == "__main__":
    unittest.main()
