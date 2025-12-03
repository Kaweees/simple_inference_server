import io
import wave

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.base import SpeechResult, SpeechSegment

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


class DummySpeechModel:
    def __init__(self) -> None:
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = "cpu"

    def transcribe(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str,
        timestamp_granularity: str | None,
    ) -> SpeechResult:
        return SpeechResult(
            text="hello audio",
            language=language or "en",
            duration=1.0,
            segments=[
                SpeechSegment(id=0, start=0.0, end=1.0, text="hello audio")
            ],
        )


class DummyRegistry:
    def __init__(self, models: dict[str, object]) -> None:
        self._models = models

    def get(self, name: str) -> object:
        if name not in self._models:
            raise KeyError
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def create_app(models: dict[str, object]) -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry(models)
    return app


def test_transcription_json() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.json()["text"] == "hello audio"


def test_transcription_text_response_format() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny", "response_format": "text"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.text.strip() == "hello audio"


def test_transcription_verbose_json_includes_segments() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny", "response_format": "verbose_json"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["language"] == "en"
    assert payload["segments"][0]["text"] == "hello audio"


def test_translation_endpoint() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/translations",
        data={"model": "openai/whisper-tiny"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.json()["text"] == "hello audio"


def test_audio_model_not_found() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "missing"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_NOT_FOUND


def test_audio_capability_required() -> None:
    class NoAudioModel:
        capabilities: list[str] = ["chat-completion"]
        device = "cpu"

    client = TestClient(create_app({"text-model": NoAudioModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "text-model"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_BAD_REQUEST


def test_audio_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()

    original_save = api._save_upload

    async def tiny_save(file: UploadFile) -> tuple[str, int]:
        return await original_save(file, max_bytes=10)

    monkeypatch.setattr(api, "_save_upload", tiny_save)

    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-dummy"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_BAD_REQUEST
