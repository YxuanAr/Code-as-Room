"""
Scene Classifier
================
Lightweight scene classifier that decides which prompt variant the pipeline should use.

Strategy: hybrid (heuristic-first, LLM-fallback).
1. First use a filename keyword heuristic; return on hit;
2. On miss, invoke the LLM (vision model) once for precise classification;
3. Any failure falls back to {"scene_type": "other", ...}; never blocks the pipeline.

Output contract (written to Memory + returned dict):
{
  "scene_type": "lab" | "residential" | "office" | "industrial" | "retail" | "other",
  "confidence": 0.0-1.0,
  "reasoning": "<<=30 words>",
  "lab_subtype": "wet_chemistry" | "biology" | "pharma_gmp" | "physics" |
                 "optics_laser" | "metrology" | "electronics" | "general" | None,
  "source": "heuristic" | "llm" | "manual" | "fallback"
}

Memory write target:
  stage="meta", type="scene_type", content=<the dict above>

Downstream consumers (Stage1 / Stage3 / etc.) read it via `read_scene_type(memory)`.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(CURRENT_DIR))


ALLOWED_SCENE_TYPES = {"lab", "residential", "office", "industrial", "retail", "other"}
ALLOWED_LAB_SUBTYPES = {
    "wet_chemistry", "biology", "pharma_gmp", "physics",
    "optics_laser", "metrology", "electronics", "general"
}

LAB_FILENAME_KEYWORDS = (
    "lab", "laboratory", "labs",
    "chemistry", "chemical", "biology", "biological", "biotech",
    "pharma", "pharmaceutical", "gmp", "qc",
    "optics", "optical", "laser", "photonics",
    "metrology", "measurement",
    "cleanroom", "clean_room",
    "physics_lab", "electronics_lab",
)
RESIDENTIAL_FILENAME_KEYWORDS = (
    "bedroom", "livingroom", "living_room", "kitchen", "bathroom",
    "studio", "apartment", "dining",
)
OFFICE_FILENAME_KEYWORDS = (
    "office", "study", "workspace", "conference",
)
INDUSTRIAL_FILENAME_KEYWORDS = (
    "warehouse", "factory", "industrial", "machine_shop", "garage",
    "server_room",
)
RETAIL_FILENAME_KEYWORDS = (
    "shop", "store", "showroom", "gallery", "cafe", "café", "restaurant",
)


# ----------------------------------------------------------------------------
# Heuristic stage
# ----------------------------------------------------------------------------
def _heuristic_from_filename(image_path: str) -> Optional[Dict[str, Any]]:
    """Filename-keyword-based heuristic classification. Returns result on hit, else None."""
    if not image_path:
        return None
    stem = Path(image_path).stem.lower()

    def _hit(keywords) -> bool:
        for kw in keywords:
            if kw in stem:
                return True
        return False

    if _hit(LAB_FILENAME_KEYWORDS):
        subtype = "general"
        if "pharma" in stem or "gmp" in stem or "qc" in stem:
            subtype = "pharma_gmp"
        elif "biology" in stem or "biotech" in stem:
            subtype = "biology"
        elif "optic" in stem or "laser" in stem or "photonic" in stem:
            subtype = "optics_laser"
        elif "metrology" in stem or "measurement" in stem:
            subtype = "metrology"
        elif "electronic" in stem:
            subtype = "electronics"
        elif "physics" in stem:
            subtype = "physics"
        elif "chemistry" in stem or "chemical" in stem:
            subtype = "wet_chemistry"
        return {
            "scene_type": "lab",
            "confidence": 0.9,
            "reasoning": f"filename hits lab keywords (stem={stem!r})",
            "lab_subtype": subtype,
            "source": "heuristic",
        }
    if _hit(RESIDENTIAL_FILENAME_KEYWORDS):
        return {
            "scene_type": "residential", "confidence": 0.85,
            "reasoning": f"filename hits residential keywords (stem={stem!r})",
            "lab_subtype": None, "source": "heuristic",
        }
    if _hit(OFFICE_FILENAME_KEYWORDS):
        return {
            "scene_type": "office", "confidence": 0.8,
            "reasoning": f"filename hits office keywords (stem={stem!r})",
            "lab_subtype": None, "source": "heuristic",
        }
    if _hit(INDUSTRIAL_FILENAME_KEYWORDS):
        return {
            "scene_type": "industrial", "confidence": 0.8,
            "reasoning": f"filename hits industrial keywords (stem={stem!r})",
            "lab_subtype": None, "source": "heuristic",
        }
    if _hit(RETAIL_FILENAME_KEYWORDS):
        return {
            "scene_type": "retail", "confidence": 0.8,
            "reasoning": f"filename hits retail keywords (stem={stem!r})",
            "lab_subtype": None, "source": "heuristic",
        }
    return None


# ----------------------------------------------------------------------------
# LLM stage
# ----------------------------------------------------------------------------
def _encode_image(image_path: str) -> Dict[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    return {"b64": b64, "mime": mime}


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = fence_match.group(1) if fence_match else text
    obj_match = re.search(r"\{[\s\S]*\}", candidate)
    if not obj_match:
        return None
    try:
        return json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return None


def _llm_classify(
    image_path: str,
    *,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Invoke the LLM for precise classification. Returns None on failure."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"[scene_classifier] langchain unavailable: {exc}")
        return None

    prompt_path = CURRENT_DIR.parent / "agent_prompt" / "Stage_scene_classifier"
    if not prompt_path.exists():
        if verbose:
            print(f"[scene_classifier] prompt does not exist: {prompt_path}")
        return None
    system_prompt = prompt_path.read_text(encoding="utf-8")

    img = _encode_image(image_path)

    llm_kwargs = dict(
        model=model,
        temperature=0.0,
        timeout=120,
        request_timeout=120,
        max_retries=2,
    )
    if base_url:
        llm_kwargs["base_url"] = base_url
    if api_key:
        llm_kwargs["api_key"] = api_key

    try:
        llm = ChatOpenAI(**llm_kwargs)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url",
                 "image_url": {"url": f"data:{img['mime']};base64,{img['b64']}"}},
                {"type": "text", "text": "Classify this room. Return JSON only."},
            ])
        ]
        response = llm.invoke(messages)
        parsed = _extract_json_from_text(getattr(response, "content", "") or "")
    except Exception as exc:
        if verbose:
            print(f"[scene_classifier] LLM call failed: {exc}")
        return None

    if not parsed:
        return None

    scene_type = str(parsed.get("scene_type", "")).strip().lower()
    if scene_type not in ALLOWED_SCENE_TYPES:
        return None
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(parsed.get("reasoning", "")).strip()[:200]
    lab_subtype = parsed.get("lab_subtype")
    if scene_type == "lab":
        if not isinstance(lab_subtype, str) or lab_subtype not in ALLOWED_LAB_SUBTYPES:
            lab_subtype = "general"
    else:
        lab_subtype = None

    return {
        "scene_type": scene_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "lab_subtype": lab_subtype,
        "source": "llm",
    }


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def classify_scene(
    image_path: str,
    *,
    memory=None,
    manual_override: Optional[str] = None,
    use_llm: bool = True,
    model: str = "gemini-3-flash-preview-thinking",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Classify the image and (optionally) write the result to Memory.

    Args:
        image_path: Input image path.
        memory: Optional Memory instance; if not None, the result is written to stage="meta".
        manual_override: User-specified scene_type; if non-empty, skips heuristic + LLM.
        use_llm: Whether to invoke the LLM when the heuristic misses.
        model/base_url/api_key: LLM configuration.
        verbose: Whether to print logs.

    Returns:
        scene_type info dict (see module docstring for the contract).
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"🏷️  [scene_classifier] {msg}")

    result: Optional[Dict[str, Any]] = None

    if manual_override:
        st = manual_override.strip().lower()
        if st in ALLOWED_SCENE_TYPES:
            result = {
                "scene_type": st,
                "confidence": 1.0,
                "reasoning": "manual override via CLI / API",
                "lab_subtype": "general" if st == "lab" else None,
                "source": "manual",
            }
            _log(f"using manually specified scene_type={st}")

    if result is None and image_path and os.path.exists(image_path):
        heur = _heuristic_from_filename(image_path)
        if heur:
            result = heur
            _log(f"heuristic hit -> scene_type={result['scene_type']} "
                 f"(confidence={result['confidence']})")

    if result is None and use_llm and image_path and os.path.exists(image_path):
        _log("heuristic missed, invoking LLM classifier...")
        llm_res = _llm_classify(
            image_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            verbose=verbose,
        )
        if llm_res:
            result = llm_res
            _log(f"LLM decision -> scene_type={result['scene_type']} "
                 f"(confidence={result['confidence']:.2f}); "
                 f"reason: {result['reasoning']}")

    if result is None:
        result = {
            "scene_type": "other",
            "confidence": 0.0,
            "reasoning": "all classification paths failed; falling back to 'other'",
            "lab_subtype": None,
            "source": "fallback",
        }
        _log("classification failed, using fallback scene_type=other")

    if memory is not None:
        try:
            memory.add(
                stage="meta",
                type="scene_type",
                content=result,
                metadata={
                    "title": f"scene_type={result['scene_type']}",
                    "summary": result.get("reasoning", "")[:80],
                    "source": result.get("source", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                },
                tags=["meta", "scene_type", result["scene_type"]],
            )
        except Exception as exc:
            _log(f"writing to Memory failed: {exc}")

    return result


def read_scene_type(memory) -> Dict[str, Any]:
    """Read the most recent scene classification result from Memory; returns fallback if not found.

    Downstream stages (Stage1 / Stage3 / etc.) call this function to determine prompt routing.
    """
    fallback = {
        "scene_type": "other",
        "confidence": 0.0,
        "reasoning": "no scene_type in memory",
        "lab_subtype": None,
        "source": "fallback",
    }
    if memory is None:
        return fallback
    try:
        latest = memory.get_latest("meta", "scene_type")
    except Exception:
        return fallback
    if not latest:
        return fallback
    content = latest.content if hasattr(latest, "content") else None
    if not isinstance(content, dict):
        return fallback
    st = str(content.get("scene_type", "other")).lower()
    if st not in ALLOWED_SCENE_TYPES:
        st = "other"
    return {
        "scene_type": st,
        "confidence": float(content.get("confidence", 0.0) or 0.0),
        "reasoning": str(content.get("reasoning", "") or ""),
        "lab_subtype": content.get("lab_subtype"),
        "source": str(content.get("source", "unknown")),
    }


def is_lab(memory, *, min_confidence: float = 0.5) -> bool:
    """Convenience check: whether the current scene is marked as a lab."""
    info = read_scene_type(memory)
    return info["scene_type"] == "lab" and info["confidence"] >= min_confidence


__all__ = [
    "classify_scene",
    "read_scene_type",
    "is_lab",
    "ALLOWED_SCENE_TYPES",
    "ALLOWED_LAB_SUBTYPES",
]
