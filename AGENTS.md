## Agent notes for this repo

### Python & tooling

- **Environment manager**: This project is managed with **uv**, not plain `python` / `pip` / `venv`.
- **Python version**: Respect `requires-python` from `pyproject.toml` (currently `>=3.12`).

### How to run things as an agent

- **Tests**: Prefer

  ```bash
  uv run pytest tests
  ```

  instead of calling `pytest` or `python -m pytest` directly.

- **One-off Python commands / scripts**:

  ```bash
  uv run python -m pytest tests/test_embedding_batching.py
  uv run python scripts/manual_smoke.py --base-url http://localhost:8000
  ```

- **Scripts with `uv` shebangs** (like `scripts/manual_smoke.py`) can be executed directly, but using `uv run ...` explicitly is also fine.

### General guidelines for tools / commands

- When constructing shell commands, **prefix Python invocations with `uv run`** unless there is a strong reason not to.
- Assume dependencies come from `pyproject.toml` / `uv.lock`; avoid installing packages with `pip`.
- Prefer absolute paths rooted at the workspace (e.g. `/path/to/simple_inference_server/...`) when calling tools.


