"""Thin LLM wrapper: throttling, retries, JSON repair. Providers: gemini, mock."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from .config import Config


class LLMError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in prose / code fences."""
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                cleaned = re.sub(r",\s*([}\]])", r"\1", text[i : j + 1])
                try:
                    return json.loads(cleaned)
                except Exception:
                    continue
    raise ValueError("no valid JSON found")


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.provider = cfg.llm_provider
        self.calls = 0
        self.failures = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self._min_interval = 0.0 if self.provider == "mock" else 60.0 / max(1, cfg.requests_per_minute)
        self._genai = None
        if self.provider == "gemini":
            if not cfg.gemini_api_key:
                raise LLMError(
                    "GEMINI_API_KEY is not set. Put it in .env (see .env.example) "
                    "or run with LLM_PROVIDER=mock for an offline demo."
                )
            import google.generativeai as genai

            genai.configure(api_key=cfg.gemini_api_key)
            self._genai = genai
        elif self.provider != "mock":
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}' (use gemini or mock)")

    def _throttle(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            wait = self._min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def complete(self, prompt: str, system: str = "", temperature: float = 0.3,
                 task: str = "generic", json_mode: bool = False) -> str:
        prompt = prompt[: self.cfg.max_prompt_chars]
        self.calls += 1
        if self.provider == "mock":
            from .mock_llm import mock_response

            return mock_response(task, prompt)

        last_err: Exception | None = None
        for attempt in range(5):
            self._throttle()
            try:
                gen_cfg: dict[str, Any] = {"temperature": temperature}
                if json_mode:
                    gen_cfg["response_mime_type"] = "application/json"
                model = self._genai.GenerativeModel(self.cfg.gemini_model, system_instruction=system or None)
                resp = model.generate_content(prompt, generation_config=gen_cfg)
                text = resp.text
                if not text:
                    raise LLMError("empty response")
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.failures += 1
                msg = str(e).lower()
                transient = any(k in msg for k in ("429", "quota", "rate", "unavailable", "503", "500", "deadline", "timeout", "empty"))
                if not transient and attempt >= 1:
                    break
                time.sleep(min(60, 5 * (2**attempt)))
        raise LLMError(f"LLM call failed ({task}): {last_err}")

    def complete_json(self, prompt: str, system: str = "", task: str = "generic",
                      temperature: float = 0.2, default: Any = None) -> Any:
        err: Exception | None = None
        for attempt in range(3):
            p = prompt if attempt == 0 else prompt + "\n\nIMPORTANT: respond with ONLY valid JSON, no commentary, no code fences."
            try:
                raw = self.complete(p, system=system, temperature=temperature, task=task, json_mode=True)
                return parse_json_loose(raw)
            except LLMError as e:
                err = e
                break
            except Exception as e:  # parse error -> retry
                err = e
        if default is not None:
            return default
        raise LLMError(f"could not get JSON for task '{task}': {err}")
