
import pytest

from app import main


def test_startup_requires_models_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure both new and legacy env vars are absent
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.delenv("MODEL_NAMES", raising=False)

    with pytest.raises(SystemExit):
        main.startup()
