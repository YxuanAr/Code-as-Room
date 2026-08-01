"""Provider configuration helpers for chat-compatible LLM endpoints."""
import os
from copy import deepcopy
from typing import Any, Dict, Optional


MINIMAX_PROVIDER: Dict[str, Any] = {
    "name": "MiniMax",
    "model_id": "MiniMax-M3",
    "model_ids": ["MiniMax-M3", "MiniMax-M2.7"],
    "models": {
        "MiniMax-M3": {
            "context_window": 1000000,
            "pricing_usd_per_million_tokens": {
                "input": 0.6,
                "output": 2.4,
                "cache_read": 0.12,
                "cache_write": None,
            },
            "input_modalities": ["text", "image", "video"],
            "thinking": ["adaptive", "disabled"],
        },
        "MiniMax-M2.7": {
            "context_window": 204800,
            "pricing_usd_per_million_tokens": {
                "input": 0.3,
                "output": 1.2,
                "cache_read": 0.06,
                "cache_write": 0.375,
            },
            "input_modalities": ["text"],
            "thinking": ["always_on"],
        },
    },
    "endpoints": {
        "global_en": {
            "openai_base_url": "https://api.minimax.io/v1",
            "anthropic_base_url": "https://api.minimax.io/anthropic",
            "docs_root": "https://platform.minimax.io/docs",
        },
        "cn_zh": {
            "openai_base_url": "https://api.minimaxi.com/v1",
            "anthropic_base_url": "https://api.minimaxi.com/anthropic",
            "docs_root": "https://platform.minimaxi.com/docs",
        },
    },
}


def get_minimax_provider_config() -> Dict[str, Any]:
    return deepcopy(MINIMAX_PROVIDER)


def is_minimax_model(model: Optional[str]) -> bool:
    return bool(model) and model in MINIMAX_PROVIDER["model_ids"]


def _minimax_region() -> str:
    region = os.environ.get("SCENEGEN_MINIMAX_REGION", "global_en").strip()
    if region not in MINIMAX_PROVIDER["endpoints"]:
        return "global_en"
    return region


def minimax_openai_base_url(region: Optional[str] = None) -> str:
    selected = region or _minimax_region()
    endpoints = MINIMAX_PROVIDER["endpoints"]
    if selected not in endpoints:
        selected = "global_en"
    return endpoints[selected]["openai_base_url"]


def supports_vision_model(model: Optional[str]) -> bool:
    if not is_minimax_model(model):
        return True
    details = MINIMAX_PROVIDER["models"].get(model or "", {})
    return "image" in details.get("input_modalities", [])


def resolve_chat_config(
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    if not is_minimax_model(model):
        return {"model": model, "base_url": base_url, "api_key": api_key}

    return {
        "model": model,
        "base_url": base_url or minimax_openai_base_url(),
        "api_key": (
            api_key
            or os.environ.get("SCENEGEN_MINIMAX_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
        ),
    }
