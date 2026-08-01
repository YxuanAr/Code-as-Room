import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AGENT_UTILS = ROOT / "agent_utils"
STAGE3 = AGENT_UTILS / "stage3"
if str(AGENT_UTILS) not in sys.path:
    sys.path.insert(0, str(AGENT_UTILS))
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MiniMaxProviderConfigTest(unittest.TestCase):
    def setUp(self):
        self.provider_config = importlib.import_module("provider_config")

    def test_resolves_global_endpoint_for_chat_config(self):
        with patch.dict("os.environ", {}, clear=True):
            resolved = self.provider_config.resolve_chat_config("MiniMax-M3")

        self.assertEqual(resolved["model"], "MiniMax-M3")
        self.assertEqual(resolved["base_url"], "https://api.minimax.io/v1")
        self.assertIsNone(resolved["api_key"])

    def test_resolves_cn_endpoint_from_region(self):
        with patch.dict("os.environ", {"SCENEGEN_MINIMAX_REGION": "cn_zh"}, clear=True):
            resolved = self.provider_config.resolve_chat_config("MiniMax-M3")

        self.assertEqual(resolved["base_url"], "https://api.minimaxi.com/v1")

    def test_records_vision_support_for_current_model(self):
        self.assertTrue(self.provider_config.supports_vision_model("MiniMax-M3"))
        self.assertFalse(self.provider_config.supports_vision_model("MiniMax-M2.7"))

    def test_stage3_payload_preserves_image_url_content(self):
        langchain_openai = types.ModuleType("langchain_openai")

        class ChatOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        langchain_openai.ChatOpenAI = ChatOpenAI
        langchain_core = types.ModuleType("langchain_core")
        langchain_messages = types.ModuleType("langchain_core.messages")

        class HumanMessage:
            def __init__(self, content):
                self.content = content

        class SystemMessage:
            def __init__(self, content):
                self.content = content

        langchain_messages.HumanMessage = HumanMessage
        langchain_messages.SystemMessage = SystemMessage

        with patch.dict(
            sys.modules,
            {
                "langchain_openai": langchain_openai,
                "langchain_core": langchain_core,
                "langchain_core.messages": langchain_messages,
            },
        ):
            sys.modules.pop("core", None)
            core = importlib.import_module("core")
            client = core.LLMClient(model="MiniMax-M3", api_key="test-key")
            payload = client._build_payload(
                [
                    SystemMessage("system"),
                    HumanMessage(
                        [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                            {"type": "text", "text": "describe"},
                        ]
                    ),
                ],
                vision=client._supports_vision(client.model),
            )

        self.assertEqual(payload["model"], "MiniMax-M3")
        self.assertEqual(client.base_url, "https://api.minimax.io/v1")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["url"], "data:image/png;base64,AAAA")


if __name__ == "__main__":
    unittest.main()
