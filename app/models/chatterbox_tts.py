from __future__ import annotations

import logging
import threading
from pathlib import Path
from chatterbox import ChatterboxTTS as ChatterboxTurboTTS

import torch

from app.models.base import TTSModel
from app.utils.device import resolve_torch_device
from app.utils.env import get_token

logger = logging.getLogger(__name__)

hf_token = get_token("HF_TOKEN")


class ChatterboxTTS(TTSModel):
    """Chatterbox Turbo TTS handler for text-to-speech generation."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.hf_repo_id = hf_repo_id
        self.name = hf_repo_id
        self.capabilities = ["text-to-speech"]
        self.device = resolve_torch_device(device, validate=False)
        # Serialize access to the underlying model
        self._lock = threading.Lock()
        self.thread_safe = False

        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        cache_dir = str(models_dir) if models_dir.exists() else get_token("HF_HOME")

        # Load the Turbo model
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        self.model = ChatterboxTurboTTS.from_pretrained(device=device_str)
        self.sr = self.model.sr

    def generate(
        self,
        text: str,
        *,
        audio_prompt_path: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> torch.Tensor:
        """Generate audio from text.

        Args:
            text: Text to convert to speech. Supports paralinguistic tags like [chuckle].
            audio_prompt_path: Optional path to a reference audio clip for voice cloning.
            cancel_event: Optional threading event to signal cancellation.

        Returns:
            Audio tensor with shape (samples,) at the model's sample rate.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS generation cancelled")

        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TTS generation cancelled")

            if audio_prompt_path is None:
                raise ValueError("audio_prompt_path is required for Chatterbox Turbo TTS")

            wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TTS generation cancelled")

            return wav

