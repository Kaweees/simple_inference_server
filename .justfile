# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias i := install
alias u := update
alias p := pre_commit
alias d := download
alias r := run
alias ch := check
alias c := clean
alias f := format

# Default command when 'just' is run without arguments
default:
  @just --list

# Install the virtual environment and pre-commit hooks
install:
  @echo "Installing..."
  @uv sync
  @uv run pre-commit install --install-hooks

update:
  @echo "Updating..."
  @uv sync --upgrade
  @uv run pre-commit autoupdate

# Run pre-commit
pre_commit:
  @echo "Running pre-commit..."
  @uv run pre-commit run -a

# Download models
download models="BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507":
  @echo "Downloading models..."
  @MODELS={{models}} uv run python scripts/download_models.py

# Run the server
run models="BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507":
  @echo "Running the server..."
  @MODELS={{models}} uv run python scripts/run_dev.py --device auto

# Run code quality tools
check:
  @echo "Checking..."
  @uv lock --locked
  @uv run pre-commit run -a

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @find . -type d -name ".venv" -exec rm -rf {} +
  @find . -type d -name "__pycache__" -exec rm -rf {} +
  @find . -type d -name "*.ruff_cache" -exec rm -rf {} +
  @find . -type d -name "*.egg-info" -exec rm -rf {} +

# Format the project
format:
  @echo "Formatting..."
  @find . -name "*.nix" -type f -exec nixfmt {} \;
