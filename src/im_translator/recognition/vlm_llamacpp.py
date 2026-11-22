"""Multimodal VLM recognizer backed by an OpenAI-compatible llama-server / Ollama endpoint."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from im_translator.core.pack_loader import LoadedPack
from im_translator.core.schemas import RecognitionOutput, RecognizerBackend
from im_translator.recognition.base import TextRecognizer
from im_translator.recognition.onnx_ocr import crop_to_png_base64


class VLMLlamaCppRecognizer(TextRecognizer):
    """Send dewarped crops to a multimodal OpenAI-compatible `/v1/chat/completions` server."""

    backend_name = RecognizerBackend.VLM_LLAMACPP

    def __init__(
        self,
        base_url: str,
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

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def recognize(self, crops: list[np.ndarray], pack: LoadedPack) -> list[RecognitionOutput]:
        if not crops:
            return []

        prompt = pack.manifest.recognition.vlm_prompt
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        outputs: list[RecognitionOutput] = []
        client_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        with httpx.Client(**client_kwargs) as client:
            for crop in crops:
                b64 = crop_to_png_base64(crop)
                payload = {
                    "model": self.model,
                    "temperature": 0.0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                }
                resp = client.post(self._endpoint(), json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                outputs.append(
                    RecognitionOutput(
                        text=text,
                        confidence=None,
                        backend=self.backend_name,
                        raw_metadata={"endpoint": self._endpoint(), "model": self.model},
                    )
                )
        return outputs
