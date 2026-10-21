"""Stage3 Base Agent - base class for all agents"""
import os
import json
import base64
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class PromptManager:
    """Prompt manager"""

    def __init__(self, prompt_dir: Optional[str] = None):
        if prompt_dir is None:
            # Default to the agent_prompt directory
            prompt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "agent_prompt"
            )
        self.prompt_dir = prompt_dir
        self._cache: Dict[str, str] = {}

    def get(self, name: str) -> str:
        """Get prompt content"""
        if name in self._cache:
            return self._cache[name]

        path = os.path.join(self.prompt_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        self._cache[name] = content
        return content

    def format(self, name: str, **kwargs) -> str:
        """Get a prompt and format it"""
        template = self.get(name)
        return template.format(**kwargs)


class LLMClient:
    """LLM client wrapper"""

    DEFAULT_MODEL = "[L]gemini-3.1-pro-preview"
    DEFAULT_BASE_URL = os.environ.get("SCENEGEN_BASE_URL")
    DEFAULT_API_KEY = os.environ.get("SCENEGEN_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # Native OpenAI model configuration
    OPENAI_MODELS = {
        "gpt-5.5",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
        "gpt-5.1",
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini",
    }
    OPENAI_BASE_URL = "https://api.openai.com/v1"

    # Reasoning model prefixes: these models don't accept the `temperature` field,
    # and often place the answer in `reasoning_content` / structured content arrays,
    # so they need compatibility handling.
    REASONING_MODEL_PREFIXES = (
        "gpt-5",
        "gpt-5.1",
        "gpt-5.5",
        "o1",
        "o3",
        "o4",
    )

    # Prefixes of text-only reasoning models known not to support vision (image_url multimodal).
    # When the fallback HTTP path lands on these models, it auto-degrades list-of-dicts
    # content into plain text to avoid 4xx errors from the gateway.
    NON_VISION_MODEL_PREFIXES = (
        "o1-mini",
        "o3-mini",
        "o4-mini",
    )

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.3
    ):
        model = model or self.DEFAULT_MODEL

        # Auto-detect whether this is a native OpenAI model
        is_openai_model = any(model.startswith(m) for m in self.OPENAI_MODELS) or model in self.OPENAI_MODELS

        # If it is an OpenAI model and base_url is not given, use the official OpenAI API
        if is_openai_model and base_url is None:
            base_url = self.OPENAI_BASE_URL
            # If api_key is not specified, try to read it from the environment
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Using OpenAI model '{model}' requires the OPENAI_API_KEY environment variable, "
                        f"or pass it via the api_key argument."
                    )
        
        common_kw = dict(
            model=model,
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key or self.DEFAULT_API_KEY,
            timeout=600,
            request_timeout=600,
            max_retries=3,
        )
        if self._is_reasoning_model(model):
            self.llm = ChatOpenAI(**common_kw)
        else:
            self.llm = ChatOpenAI(**common_kw, temperature=temperature)
        
        self.model = model
        self.base_url = base_url or (self.OPENAI_BASE_URL if is_openai_model else self.DEFAULT_BASE_URL)
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.temperature = temperature
        self.timeout = 600

    @staticmethod
    def _is_langchain_response_shape_error(err_str: str) -> bool:
        return "model_dump" in err_str and "object has no attribute" in err_str

    @classmethod
    def _is_reasoning_model(cls, model: str) -> bool:
        """Reasoning models don't accept the `temperature` field, and use a special response format."""
        if not model:
            return False
        return any(model.startswith(p) for p in cls.REASONING_MODEL_PREFIXES)

    @classmethod
    def _supports_vision(cls, model: str) -> bool:
        """Conservative check: text-only mini / text-only reasoning models default to no vision."""
        if not model:
            return True
        return not any(model.startswith(p) for p in cls.NON_VISION_MODEL_PREFIXES)

    @staticmethod
    def _flatten_content_to_text(content: Any) -> str:
        """Flatten LangChain vision-style list-of-dicts content into plain text.

        Drops the image_url segments (used by fallbacks where the gateway has no image
        support); keeps only the text segments.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("type")
                    if t == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif t in (None, "image_url"):
                        # image_url segments are dropped; skip other unknown types
                        continue
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p)
        return str(content) if content is not None else ""

    @classmethod
    def _message_to_openai_dict(cls, message: Any, *, vision: bool = True) -> Dict[str, Any]:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
            if not vision:
                content = cls._flatten_content_to_text(content)
            return {"role": role, "content": content}

        content = getattr(message, "content", str(message))
        class_name = message.__class__.__name__.lower()
        if "system" in class_name:
            role = "system"
        elif "ai" in class_name or "assistant" in class_name:
            role = "assistant"
        else:
            role = "user"
        if not vision:
            content = cls._flatten_content_to_text(content)
        return {"role": role, "content": content}

    @staticmethod
    def _coerce_message_field(value: Any) -> str:
        """Forcibly coerce a message field (content / reasoning_content / ...) into text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
                    elif isinstance(item.get("value"), str):
                        parts.append(item["value"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p)
        if isinstance(value, dict):
            for k in ("text", "content", "value"):
                v = value.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            return ""
        return str(value)

    @classmethod
    def _extract_openai_content(cls, data: Any) -> str:
        """Extract visible text from an OpenAI-compatible response.

        Handles reasoning models that put the answer in non-standard fields like
        `reasoning_content` / `reasoning.content` / top-level `output_text` (many
        proxy gateways do this).
        """
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return str(data)

        # 1) Standard chat.completions: choices[0].message.content
        choices = data.get("choices") or []
        message: Dict[str, Any] = {}
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = cls._coerce_message_field(message.get("content"))
            if text and text.strip():
                return text

            # 2) Reasoning fallback: reasoning_content / reasoning.content
            for key in ("reasoning_content", "reasoning"):
                v = message.get(key)
                if isinstance(v, dict):
                    v = v.get("content", v)
                text = cls._coerce_message_field(v)
                if text and text.strip():
                    return text

            # 3) Some gateways put text at choices[0].text
            text = cls._coerce_message_field(choices[0].get("text"))
            if text and text.strip():
                return text

            # 4) tool_calls / function_call - rare fallback
            tcs = message.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                for tc in tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        args = fn.get("arguments")
                        if isinstance(args, str) and args.strip():
                            return args

        # 5) Responses-API style: top-level output_text or output[*].content[*].text
        text = cls._coerce_message_field(data.get("output_text"))
        if text and text.strip():
            return text
        outputs = data.get("output")
        if isinstance(outputs, list):
            chunks: List[str] = []
            for out in outputs:
                if isinstance(out, dict):
                    c = out.get("content")
                    chunks.append(cls._coerce_message_field(c))
            joined = "\n".join(c for c in chunks if c)
            if joined.strip():
                return joined

        return ""

    def _build_payload(self, messages: List, *, vision: bool) -> Dict[str, Any]:
        """Build OpenAI-compatible chat.completions payload; reasoning models omit temperature."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_openai_dict(m, vision=vision) for m in messages],
            "stream": False,
        }
        if not self._is_reasoning_model(self.model):
            payload["temperature"] = self.temperature
        return payload

    @staticmethod
    def _dump_fallback_raw(payload: Dict[str, Any], raw_text: str) -> None:
        """Persist the fallback raw response to disk to ease post-mortem debugging. Failures don't affect the main flow."""
        try:
            out_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, "_fallback_raw.json")
            wrapped = {
                "request_payload_meta": {
                    "model": payload.get("model"),
                    "n_messages": len(payload.get("messages") or []),
                    "has_temperature": "temperature" in payload,
                },
                "raw_response_text": raw_text,
            }
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(wrapped, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _direct_chat_completion(self, messages: List) -> str:
        """Call an OpenAI-compatible chat endpoint without LangChain.

        Robustified to:
        - drop `temperature` for reasoning models (gpt-5.x / o-series)
        - retry once with vision content stripped if first call 4xx's
        - extract from non-standard response fields (reasoning_content, etc.)
        - persist raw response to disk for offline debugging
        """
        import urllib.error
        import urllib.request

        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        def _post(payload: Dict[str, Any]) -> Tuple[int, str]:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.status, resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                return exc.code, detail

        prefer_vision = self._supports_vision(self.model)
        payload = self._build_payload(messages, vision=prefer_vision)
        status, raw = _post(payload)

        if status >= 400 and prefer_vision:
            # Some gateways reject vision content arrays for text-only models.
            # Retry once with content flattened to plain text.
            print(
                f"LLMClient direct fallback HTTP {status} with vision content; "
                f"retrying with text-only payload"
            )
            payload = self._build_payload(messages, vision=False)
            status, raw = _post(payload)

        if status >= 400:
            self._dump_fallback_raw(payload, raw)
            raise RuntimeError(
                f"Direct chat completion failed: HTTP {status}: "
                f"{raw[:500]}{'…' if len(raw) > 500 else ''}"
            )

        # Persisting successful responses also helps with replay
        self._dump_fallback_raw(payload, raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        content = self._extract_openai_content(data)
        if not content or not str(content).strip():
            # Provide a readable diagnostic upstream
            try:
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                keys = sorted(list(msg.keys())) if isinstance(msg, dict) else []
                top_keys = sorted(list(data.keys())) if isinstance(data, dict) else []
                print(
                    f"LLMClient direct fallback got empty visible content "
                    f"(top_keys={top_keys}, message_keys={keys}); "
                    f"raw saved to stage3/output/_fallback_raw.json"
                )
            except Exception:
                pass
        return content
    
    def invoke(self, messages: List) -> str:
        """Call the LLM and return the text content (auto-retry on transient errors).

        Retries on:
            - HTTP 502/503/429 / connection / timeout errors (str-matched)
            - HTTP 200 with empty / whitespace-only content. This is the
              gemini-3-flash-thinking failure mode where reasoning tokens
              consume the visible-answer budget; the gateway returns 200
              but content == "". Without this branch every empty response
              would silently kill a stage (Stage 4 in particular, which
              has no fallback otherwise).
        After exhausting all retries on empty content, returns "" so the
        caller can choose to skip / degrade gracefully rather than raise.
        """
        import time as _time
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)
                if content is None or not str(content).strip():
                    if attempt < max_retries:
                        wait = min(10 * attempt, 60)
                        print(
                            f"LLMClient empty content (attempt {attempt}/{max_retries}) "
                            f"- retry in {wait}s"
                        )
                        _time.sleep(wait)
                        continue
                    print(
                        f"LLMClient empty content after {max_retries} attempts; "
                        f"returning empty string (caller may degrade)."
                    )
                    return ""
                return content
            except Exception as e:
                err_str = str(e)
                if self._is_langchain_response_shape_error(err_str):
                    print("LLMClient LangChain response parsing failed; using direct HTTP fallback")
                    content = self._direct_chat_completion(messages)
                    if content is None or not str(content).strip():
                        if attempt < max_retries:
                            wait = min(10 * attempt, 60)
                            print(
                                f"LLMClient direct fallback empty content "
                                f"(attempt {attempt}/{max_retries}) - retry in {wait}s"
                            )
                            _time.sleep(wait)
                            continue
                        return ""
                    return content
                is_retryable = any(k in err_str for k in ("502", "503", "429", "Connection error", "upstream", "timeout", "Timeout"))
                if is_retryable and attempt < max_retries:
                    wait = min(10 * attempt, 60)
                    print(f"LLMClient API error (attempt {attempt}/{max_retries}): {e} - retry in {wait}s")
                    _time.sleep(wait)
                    continue
                raise


class BaseAgent(ABC):
    """Agent base class - defines the basic structure of an Agent"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        verbose: bool = True
    ):
        self.llm = llm_client or LLMClient()
        self.prompts = prompt_manager or PromptManager()
        self.verbose = verbose

    @property
    @abstractmethod
    def system_prompt_name(self) -> str:
        """System prompt filename"""
        pass

    @property
    @abstractmethod
    def user_prompt_name(self) -> str:
        """User prompt template filename"""
        pass

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {
                "info": "[i]",
                "success": "[OK]",
                "warning": "[!]",
                "error": "[X]"
            }.get(level, "")
            print(f"{prefix} {msg}")

    def get_system_prompt(self) -> str:
        """Get the system prompt"""
        return self.prompts.get(self.system_prompt_name)

    def build_user_prompt(self, **kwargs) -> str:
        """Build the user prompt - subclasses may override"""
        return self.prompts.format(self.user_prompt_name, **kwargs)

    def build_messages(self, **kwargs) -> List:
        """Build the message list - subclasses may override to support images"""
        system = self.get_system_prompt()
        user = self.build_user_prompt(**kwargs)

        return [
            SystemMessage(content=system),
            HumanMessage(content=user)
        ]

    def invoke(self, **kwargs) -> str:
        """Call the LLM"""
        messages = self.build_messages(**kwargs)
        return self.llm.invoke(messages)

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Run the Agent task - subclasses must implement"""
        pass


class ImageMixin:
    """Image-processing mixin class"""

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode an image as base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_mime_type(image_path: str) -> str:
        """Get an image MIME type"""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(ext, "image/png")

    def build_image_content(self, image_path: str) -> Dict:
        """Build image message content"""
        b64 = self.encode_image(image_path)
        mime = self.get_mime_type(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        }


def extract_json_from_response(text: str) -> str:
    """Extract JSON from an LLM response"""
    import re

    # Try to find a ```json ... ``` block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        return match.group(1).strip()

    # Try to find {...} or [...]
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if match:
        return match.group(1).strip()

    return text.strip()


def _strip_markdown_fences(code: str) -> str:
    """Remove any remaining markdown code fence markers."""
    import re
    lines = code.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^```\w*$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def extract_python_from_response(text: str) -> str:
    """Extract Python code from LLM response, handling truncated blocks."""
    import re
    
    if not text or not text.strip():
        return ""
    
    code = ""
    
    # Complete code block: ```python ... ```
    python_match = re.search(r'```python\s*([\s\S]*?)\s*```', text)
    if python_match:
        c = python_match.group(1).strip()
        if c and ('import' in c or 'def ' in c or 'bpy' in c):
            code = c
    
    # Any complete code block containing bpy
    if not code:
        all_blocks = re.findall(r'```(?:\w*)\s*([\s\S]*?)\s*```', text)
        for block in all_blocks:
            block = block.strip()
            if 'import bpy' in block or ('bpy.' in block and 'import' in block):
                code = block
                break
    
    # Any complete code block
    if not code:
        match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', text)
        if match:
            c = match.group(1).strip()
            if c and ('import' in c or 'def ' in c or 'bpy' in c):
                code = c
    
    # Truncated code block: opening ``` without closing (LLM hit token limit)
    if not code:
        trunc_match = re.search(r'```(?:python)?\s*([\s\S]+)', text)
        if trunc_match:
            c = trunc_match.group(1).strip()
            if c and ('import' in c or 'def ' in c or 'bpy' in c):
                code = c
    
    # Raw code without fences
    if not code:
        stripped = text.strip()
        if stripped.startswith('import'):
            code = stripped
        elif 'import' in stripped or 'def ' in stripped or 'bpy' in stripped:
            code = stripped
    
    # Safety net: strip any remaining fence markers regardless of extraction path
    if code:
        code = _strip_markdown_fences(code)
    
    return code
