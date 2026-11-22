"""OpenAI-compatible HTTP client for llama-server, Ollama, vLLM, and cloud APIs."""

from __future__ import annotations

from typing import Any

import httpx


class OpenAICompatibleClient:
    """Thin httpx client for OpenAI-compatible `/v1/chat/completions` and `/v1/models`."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "not-needed",
        model: str = "local-model",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._transport = transport

    def _url(self, suffix: str) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}{suffix}"
        return f"{self.base_url}/v1{suffix}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def health_check(self) -> dict[str, Any]:
        """Query `/v1/models` to verify connectivity and inspect available model IDs."""
        url = self._url("/models")
        try:
            with self._client() as client:
                resp = client.get(url, headers=self._headers())
                resp.raise_for_status()
                payload = resp.json()
            models = [m.get("id", "") for m in payload.get("data", []) if isinstance(m, dict)]
            return {"ok": True, "url": url, "models": models}
        except Exception as exc:
            return {"ok": False, "url": url, "models": [], "error": str(exc)}

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> str:
        """Send a chat completion request and return assistant text content."""
        payload = {
            "model": model or self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        with self._client() as client:
            resp = client.post(self._url("/chat/completions"), json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"LLM endpoint returned empty choices: {data}")
        return str(choices[0]["message"]["content"]).strip()
