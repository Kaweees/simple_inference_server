from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, Iterable

import torch
import yaml

from app.models.base import EmbeddingModel
from app.models.bge_m3 import BgeM3Embedding
from app.models.embedding_gemma import EmbeddingGemmaEmbedding


class ModelRegistry:
    def __init__(
        self,
        config_path: str,
        device: str | None = None,
        allowed_models: Iterable[str] | None = None,
    ) -> None:
        self.models: dict[str, EmbeddingModel] = {}
        # Prefer CLI/env provided device; fall back to auto-detection.
        self.device_preference: str = (device or os.getenv("MODEL_DEVICE") or "auto")
        self.device = self._resolve_device(self.device_preference)
        self.allowed_models = {m.strip() for m in allowed_models or [] if m.strip()} or None
        self._load_from_config(config_path)

    def _resolve_device(self, preference: str | None) -> str:
        pref = (preference or "auto").lower()
        has_cuda = torch.cuda.is_available()
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

        if pref == "auto":
            if has_cuda:
                return "cuda"
            if has_mps:
                return "mps"
            return "cpu"

        if pref == "cpu":
            return "cpu"

        if pref == "mps":
            if not has_mps:
                raise ValueError("MPS requested but not available")
            return "mps"

        if pref.startswith("cuda"):
            if not has_cuda:
                raise ValueError("CUDA requested but not available")

            if ":" in pref:
                _, idx_str = pref.split(":", 1)
                if not idx_str.isdigit():
                    raise ValueError(f"Invalid CUDA device format: {preference}")
                idx = int(idx_str)
                count = torch.cuda.device_count()
                if idx >= count:
                    raise ValueError(f"Requested cuda:{idx} but only {count} device(s) visible")
                return f"cuda:{idx}"

            return "cuda"

        raise ValueError(f"Unknown device preference: {preference}")

    def _load_from_config(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model config not found: {path}")

        with path_obj.open() as f:
            cfg = yaml.safe_load(f)
        models_cfg = cfg.get("models", [])
        if not models_cfg:
            raise ValueError("No models configured")

        requested = set(self.allowed_models) if self.allowed_models is not None else None
        loaded: set[str] = set()
        for item in models_cfg:
            name = item["name"]
            if requested is not None and name not in requested:
                continue
            handler_path = item.get("handler")
            repo = item["hf_repo_id"]

            if handler_path:
                handler_factory = self._import_handler(handler_path)
            else:
                handler_factory = self._default_handler_for(name)

            model = handler_factory(repo, self.device)

            self.models[name] = model
            loaded.add(name)

        if requested is not None:
            missing = requested - loaded
            if missing:
                raise ValueError(f"Requested model(s) not found in config: {', '.join(sorted(missing))}")

    def _import_handler(self, dotted_path: str) -> Callable[[str, str], EmbeddingModel]:
        if "." not in dotted_path:
            raise ValueError(f"Handler path must be module.Class, got: {dotted_path}")
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        try:
            handler = getattr(module, class_name)
            return handler
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ImportError(f"Handler class {class_name} not found in {module_path}") from exc

    def _default_handler_for(self, name: str) -> Callable[[str, str], EmbeddingModel]:
        if name == "bge-m3":
            return BgeM3Embedding
        if name == "embedding-gemma-300m":
            return EmbeddingGemmaEmbedding
        raise ValueError(f"Unknown model name: {name}")

    def get(self, name: str) -> EmbeddingModel:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not loaded")
        return self.models[name]

    def list_models(self) -> list[str]:
        return list(self.models.keys())
