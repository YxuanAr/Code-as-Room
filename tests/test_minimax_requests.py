"""Offline request-boundary coverage using the real LangChain/OpenAI serializer."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import httpx2 as httpx
except ImportError:
    import httpx
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent_utils"))

from provider_config import get_minimax_provider_config, resolve_chat_config


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_source("request_test_core", "agent_utils/stage3/core.py")


class MiniMaxRequestTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        network = patch("socket.socket.connect", side_effect=AssertionError("Network disabled"))
        network.start()
        self.addCleanup(network.stop)
        self.requests = []
        self.response_text = "room"
        self.content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Describe the room."},
        ]
        self.messages = [SystemMessage(content="system"), HumanMessage(content=self.content)]
        self.default_model = get_minimax_provider_config()["model_id"]

        def send(client, request, **kwargs):
            self.requests.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, request=request, json={
                "id": "offline-response", "object": "chat.completion", "created": 0,
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": self.response_text,
                }}],
            })

        transport = patch.object(httpx.Client, "send", send)
        transport.start()
        self.addCleanup(transport.stop)
        # Any unexpected direct HTTP fallback must fail rather than use the network.
        fallback = patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected HTTP"))
        fallback.start()
        self.addCleanup(fallback.stop)

    def assert_request(self, model, base_url, content=None):
        url, body = self.requests[-1]
        self.assertEqual(url, base_url + "/chat/completions")
        self.assertEqual(body["model"], model)
        self.assertEqual(body["messages"][1]["content"], self.content if content is None else content)
        self.assertFalse(body["stream"])
        for unsupported in ("thinking", "reasoning_effort", "reasoning_split", "service_tier"):
            self.assertNotIn(unsupported, body)

    def test_catalog_defaults_and_legacy_resolution(self):
        config = get_minimax_provider_config()
        self.assertEqual(config["model_id"], "MiniMax-M3.1-Flash")
        self.assertEqual(config["model_ids"], ["MiniMax-M3.1-Flash", "MiniMax-M3"])
        self.assertEqual(config["models"][self.default_model]["thinking"], ["adaptive"])
        self.assertEqual(resolve_chat_config("MiniMax-M2.7")["base_url"], "https://api.minimax.io/v1")
        self.assertIsNone(resolve_chat_config(None)["model"])

    def test_stage3_sdk_default_and_saved_models_in_both_regions(self):
        for region, base_url in (("global_en", "https://api.minimax.io/v1"),
                                 ("cn_zh", "https://api.minimaxi.com/v1")):
            for model in (self.default_model, "MiniMax-M3", "MiniMax-M2.7"):
                with self.subTest(region=region, model=model), patch.dict(
                    "os.environ", {"SCENEGEN_MINIMAX_REGION": region}
                ):
                    client = core.LLMClient(model=model, api_key="offline-key")
                    self.assertEqual(client.invoke(self.messages), "room")
                    self.assert_request(model, base_url)

    def test_explicit_custom_endpoint_and_other_models_are_preserved(self):
        for model in (self.default_model, "MiniMax-M3", "MiniMax-M2.7", "custom-model", "gpt-4o"):
            with self.subTest(model=model):
                client = core.LLMClient(model=model, base_url="https://custom.invalid/v1",
                                        api_key="offline-key")
                self.assertEqual(client.invoke(self.messages), "room")
                self.assert_request(model, "https://custom.invalid/v1")

    def test_non_minimax_default_is_unchanged(self):
        client = core.LLMClient(base_url="https://custom.invalid/v1", api_key="offline-key")
        self.assertEqual(client.model, "[L]gemini-3.1-pro-preview")
        self.assertEqual(client.invoke(self.messages), "room")
        self.assert_request(client.model, "https://custom.invalid/v1")

    def test_stage3_real_fallback_serializes_default_and_saved_models(self):
        for region, base_url in (("global_en", "https://api.minimax.io/v1"),
                                 ("cn_zh", "https://api.minimaxi.com/v1")):
            for model in (self.default_model, "MiniMax-M3", "MiniMax-M2.7"):
                with self.subTest(region=region, model=model), patch.dict(
                    "os.environ", {"SCENEGEN_MINIMAX_REGION": region}
                ):
                    client = core.LLMClient(model=model, api_key="offline-key")
                    response = MagicMock()
                    response.__enter__.return_value = response
                    response.status = 200
                    response.read.return_value = b'{"choices":[{"message":{"content":"room"}}]}'
                    with patch.object(type(client.llm), "invoke", side_effect=AttributeError(
                        "object has no attribute model_dump"
                    )), patch("urllib.request.urlopen", return_value=response) as send, patch.object(
                        core.LLMClient, "_dump_fallback_raw"
                    ):
                        self.assertEqual(client.invoke(self.messages), "room")
                    request = send.call_args.args[0]
                    self.requests.append((request.full_url, json.loads(request.data)))
                    expected = "Describe the room." if model == "MiniMax-M2.7" else self.content
                    self.assert_request(model, base_url, expected)

    def test_scene_classifier_sends_catalog_default_through_sdk(self):
        classifier = load_source("request_test_classifier", "agent_utils/scene_classifier.py")
        self.response_text = json.dumps({"scene_type": "residential", "confidence": 0.9})
        with patch.object(classifier, "_encode_image", return_value={"mime": "image/png", "b64": "AAAA"}):
            result = classifier._llm_classify(
                "unused.png", model=self.default_model, base_url=None,
                api_key="offline-key", verbose=False,
            )
        self.assertEqual(result["scene_type"], "residential")
        self.assert_request(self.default_model, "https://api.minimax.io/v1", [
            self.content[0], {"type": "text", "text": "Classify this room. Return JSON only."},
        ])

    def test_stage1_and_stage2_send_catalog_default_through_sdk(self):
        for stage in (1, 2):
            module = load_source(f"request_test_stage{stage}", f"agent_utils/stage{stage}/run_stage{stage}.py")
            runner_type = getattr(module, f"Stage{stage}Runner")
            with self.subTest(stage=stage):
                runner = runner_type(image_path=str(ROOT / "example/example1.png"),
                                     model=self.default_model, api_key="offline-key", verbose=False)
                self.assertEqual(runner.llm.invoke(self.messages).content, "room")
                self.assert_request(self.default_model, "https://api.minimax.io/v1")


if __name__ == "__main__":
    unittest.main()
