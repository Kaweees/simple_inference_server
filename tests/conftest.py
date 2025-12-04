"""Test-wide fixtures and stubs."""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from typing import Any

# Several tests (and third-party imports) look for torchaudio. Provide a minimal
# stub with a ModuleSpec so importlib.util.find_spec works without the real
# package installed.
_torchaudio: Any = sys.modules.get("torchaudio", types.ModuleType("torchaudio"))
if getattr(_torchaudio, "__spec__", None) is None:
    _torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)

if not hasattr(_torchaudio, "info"):
    _torchaudio.info = lambda _path: types.SimpleNamespace(num_frames=0, sample_rate=0)

sys.modules["torchaudio"] = _torchaudio

# Provide a minimal torch stub for tests that only need to import modules.
_torch: Any = sys.modules.get("torch", types.ModuleType("torch"))
if getattr(_torch, "__spec__", None) is None:
    _torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)

_torch.cuda = getattr(_torch, "cuda", types.SimpleNamespace(is_available=lambda: False))
_torch.xpu = getattr(_torch, "xpu", None)
_torch.no_grad = getattr(_torch, "no_grad", contextlib.nullcontext)

sys.modules["torch"] = _torch

# Pillow is also imported in some modules; provide a lightweight stub.
_pil: Any = sys.modules.get("PIL", types.ModuleType("PIL"))
if getattr(_pil, "__spec__", None) is None:
    _pil.__spec__ = importlib.machinery.ModuleSpec("PIL", loader=None, is_package=True)
if not hasattr(_pil, "__path__"):
    _pil.__path__ = []

_pil_image: Any = types.ModuleType("PIL.Image")
_pil_image.__spec__ = importlib.machinery.ModuleSpec("PIL.Image", loader=None)

class _Image:  # pragma: no cover - stub type only
    pass

_pil_image.Image = _Image
_pil.Image = _Image

sys.modules["PIL"] = _pil
sys.modules["PIL.Image"] = _pil_image

# transformers is not installed in CI; provide a lightweight stub for tests that
# import modules but do not execute model logic.
_transformers: Any = sys.modules.get("transformers", types.ModuleType("transformers"))
if getattr(_transformers, "__spec__", None) is None:
    _transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

class _StoppingCriteria:  # pragma: no cover - stub type only
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

class _StoppingCriteriaList(list):  # pragma: no cover - stub type only
    pass

class _DummyProcessor:  # pragma: no cover - stub type only
    tokenizer: Any = types.SimpleNamespace(encode=lambda *a, **k: [])

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _DummyProcessor:
        return cls()

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return {"input_ids": types.SimpleNamespace(shape=(1, 0)), "to": lambda self, *_: self}

    def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
        return [""]

class _DummyModel:  # pragma: no cover - stub type only
    device: Any = "cpu"

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _DummyModel:
        return cls()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(__getitem__=lambda self, idx: self)  # minimal placeholder

    def to(self, *args: Any, **kwargs: Any) -> _DummyModel:
        return self

    def eval(self) -> None:
        return None

_transformers.AutoModelForCausalLM = _DummyModel
_transformers.AutoProcessor = _DummyProcessor
_transformers.StoppingCriteria = _StoppingCriteria
_transformers.StoppingCriteriaList = _StoppingCriteriaList
_transformers.__version__ = "0.0.0"

sys.modules["transformers"] = _transformers

# numpy is imported for type declarations; stub it to avoid pulling heavy deps in CI.
_numpy: Any = sys.modules.get("numpy", types.ModuleType("numpy"))
if getattr(_numpy, "__spec__", None) is None:
    _numpy.__spec__ = importlib.machinery.ModuleSpec("numpy", loader=None)

class _NDArray:  # pragma: no cover - stub type only
    pass

_numpy.ndarray = _NDArray
_numpy.array = getattr(_numpy, "array", lambda *a, **k: a[0] if a else None)

sys.modules["numpy"] = _numpy

