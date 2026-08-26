import json
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lib_run_single import (
    LLMRetryExhaustedError,
    RecoverableModelError,
    _is_recoverable_model_error,
    run_single_example,
)
from mm_agents.agent import PromptAgent
from mm_agents.qwen.main import QwenAgent
from PIL import Image


class FakeController:
    def start_recording(self):
        pass

    def end_recording(self, _path):
        pass


class FakeEnv:
    vm_ip = "127.0.0.1"

    def __init__(self):
        self.controller = FakeController()
        self.evaluate_calls = 0

    def reset(self, task_config):
        self.task_config = task_config

    def _get_obs(self):
        return {"screenshot": b"screenshot"}

    def step(self, _action, _pause):
        return self._get_obs(), 0.0, True, {"done": True}

    def evaluate(self):
        self.evaluate_calls += 1
        return 1.0


class FakeAgent:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = 0

    def reset(self, *_args, **_kwargs):
        pass

    def predict(self, _instruction, _obs):
        self.calls += 1
        return next(self.predictions)


class LLMRetrySemanticsTest(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "OSWORLD_LLM_MAX_RETRIES_PER_STEP": "3",
                "OSWORLD_LLM_RETRY_BACKOFF_SECONDS": "0",
            },
        )
        self.env_patch.start()
        self.sleep_patch = patch("lib_run_single.time.sleep", return_value=None)
        self.sleep_patch.start()

    def tearDown(self):
        self.sleep_patch.stop()
        self.env_patch.stop()

    def test_failures_do_not_consume_step_before_success(self):
        agent = FakeAgent([
            ("", []),
            ("", []),
            ("DONE", ["DONE"]),
        ])
        env = FakeEnv()
        args = SimpleNamespace(sleep_after_execution=0)

        with tempfile.TemporaryDirectory() as result_dir:
            with patch("lib_run_single.log_task_completion"):
                run_single_example(
                    agent,
                    env,
                    {"id": "test", "domain": "test"},
                    15,
                    "test instruction",
                    args,
                    result_dir,
                    [],
                )
            with open(os.path.join(result_dir, "traj.jsonl"), encoding="utf-8") as f:
                trajectory = json.loads(f.readline())

            self.assertEqual(agent.calls, 3)
            self.assertEqual(trajectory["step_num"], 1)
            self.assertTrue(os.path.exists(os.path.join(result_dir, "result.txt")))

    def test_exhausted_retries_do_not_evaluate_or_write_result(self):
        agent = FakeAgent([("", []), ("", []), ("", [])])
        env = FakeEnv()
        args = SimpleNamespace(sleep_after_execution=0)

        with tempfile.TemporaryDirectory() as result_dir:
            with self.assertRaises(LLMRetryExhaustedError):
                run_single_example(
                    agent,
                    env,
                    {"id": "test", "domain": "test"},
                    15,
                    "test instruction",
                    args,
                    result_dir,
                    [],
                )

            self.assertEqual(agent.calls, 3)
            self.assertEqual(env.evaluate_calls, 0)
            self.assertFalse(os.path.exists(os.path.join(result_dir, "result.txt")))

    def test_nonempty_model_failure_consumes_step_without_retry(self):
        agent = FakeAgent([("explanation without an action", [])])
        env = FakeEnv()
        args = SimpleNamespace(sleep_after_execution=0)

        with tempfile.TemporaryDirectory() as result_dir:
            with patch("lib_run_single.log_task_completion"):
                run_single_example(
                    agent,
                    env,
                    {"id": "test", "domain": "test"},
                    1,
                    "test instruction",
                    args,
                    result_dir,
                    [],
                )

            self.assertEqual(agent.calls, 1)
            self.assertEqual(env.evaluate_calls, 1)
            self.assertTrue(os.path.exists(os.path.join(result_dir, "result.txt")))

    def test_transport_exception_retries_without_consuming_step(self):
        class ReadTimeout(Exception):
            pass

        class FlakyAgent:
            calls = 0

            def reset(self, *_args, **_kwargs):
                pass

            def predict(self, _instruction, _obs):
                self.calls += 1
                if self.calls < 3:
                    raise ReadTimeout("upstream timed out")
                return "DONE", ["DONE"]

        agent = FlakyAgent()
        env = FakeEnv()
        args = SimpleNamespace(sleep_after_execution=0)

        with tempfile.TemporaryDirectory() as result_dir:
            with patch("lib_run_single.log_task_completion"):
                run_single_example(
                    agent,
                    env,
                    {"id": "test", "domain": "test"},
                    1,
                    "test instruction",
                    args,
                    result_dir,
                    [],
                )

        self.assertEqual(agent.calls, 3)
        self.assertEqual(env.evaluate_calls, 1)

    def test_transport_exception_exhaustion_is_task_recoverable(self):
        class ReadTimeout(Exception):
            pass

        class FailingAgent:
            def reset(self, *_args, **_kwargs):
                pass

            def predict(self, _instruction, _obs):
                raise ReadTimeout("upstream timed out")

        with tempfile.TemporaryDirectory() as result_dir:
            with self.assertRaises(RecoverableModelError):
                run_single_example(
                    FailingAgent(),
                    FakeEnv(),
                    {"id": "test", "domain": "test"},
                    1,
                    "test instruction",
                    SimpleNamespace(sleep_after_execution=0),
                    result_dir,
                    [],
                )

    def test_gateway_waf_html_is_recoverable_but_bad_credentials_are_not(self):
        waf_error = RuntimeError(
            "Non-JSON response from OpenAI-compatible endpoint: "
            "endpoint=http://phoenix-gw-eval.alibaba.com/eval/dashscope/chat/completions, "
            "status=403, bxpunish=true, waf_uuid=abc"
        )
        self.assertTrue(_is_recoverable_model_error(waf_error))

        class PermissionDeniedError(Exception):
            pass

        self.assertFalse(
            _is_recoverable_model_error(PermissionDeniedError("invalid API key"))
        )


class PredictionRollbackTest(unittest.TestCase):
    def test_empty_response_rolls_back_agent_history(self):
        agent = PromptAgent(
            model="qwen3.8-max",
            action_space="pyautogui",
            observation_type="screenshot",
        )
        agent.call_llm = lambda _payload: ""

        response, actions = agent.predict(
            "test instruction",
            {"screenshot": b"screenshot"},
        )

        self.assertEqual(response, "")
        self.assertEqual(actions, [])
        self.assertEqual(agent.observations, [])
        self.assertEqual(agent.actions, [])
        self.assertEqual(agent.thoughts, [])

    def test_nonempty_model_failure_is_kept_in_agent_history(self):
        agent = PromptAgent(
            model="qwen3.8-max",
            action_space="pyautogui",
            observation_type="screenshot",
        )
        agent.call_llm = lambda _payload: "explanation without a code block"

        response, actions = agent.predict(
            "test instruction",
            {"screenshot": b"screenshot"},
        )

        self.assertEqual(response, "explanation without a code block")
        self.assertEqual(actions, [])
        self.assertEqual(len(agent.observations), 1)
        self.assertEqual(len(agent.actions), 1)
        self.assertEqual(len(agent.thoughts), 1)

    def test_qwen_empty_response_rolls_back_agent_history(self):
        image = io.BytesIO()
        Image.new("RGB", (16, 16), "white").save(image, format="PNG")
        agent = QwenAgent(model="qwen3.8-max")
        agent.call_llm = lambda _payload, _model: ""

        response, actions = agent.predict(
            "test instruction",
            {"screenshot": image.getvalue()},
        )

        self.assertEqual(response, "")
        self.assertEqual(actions, [])
        self.assertEqual(agent.screenshots, [])
        self.assertEqual(agent.observations, [])
        self.assertEqual(agent.responses, [])
        self.assertEqual(agent.actions, [])
        self.assertEqual(agent.folded_prefix_k, 0)


if __name__ == "__main__":
    unittest.main()
