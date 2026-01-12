# Installation

## Requirements

- **Python 3.12 or higher**
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Package Installation

We recommend using [uv](https://github.com/astral-sh/uv) for faster, more reliable package management:

```bash
uv pip install pitchpredict
```

Or with pip:

```bash
pip install pitchpredict
```

This installs PitchPredict and all required dependencies.

## Development Installation

For contributing or development work:

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/baseball-analytica/pitchpredict.git
cd pitchpredict

# Sync dependencies from lock file
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/baseball-analytica/pitchpredict.git
cd pitchpredict

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | REST API server framework |
| uvicorn | ASGI server |
| pandas | Data manipulation |
| numpy | Numerical computing |
| torch | Deep learning framework |
| pybaseball | MLB Statcast data fetching |
| rich | Terminal formatting |
| tqdm | Progress bars |
| wandb | Experiment tracking (for training) |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-asyncio | Async test support |
| mypy | Type checking |
| ruff | Linting and formatting |

## Verify Installation

After installation, verify it works:

```bash
# Check the CLI is available
pitchpredict --help

# Or start the server
pitchpredict serve --port 8056
```

In Python:

```python
from pitchpredict import PitchPredict
print("PitchPredict installed successfully!")
```

## GPU Support

For deep learning features, PitchPredict uses PyTorch which can leverage GPU acceleration. GPU support is optional but recommended for training models.

To use GPU:

1. Install PyTorch with CUDA support per [PyTorch installation guide](https://pytorch.org/get-started/locally/)
2. PitchPredict will automatically detect and use the GPU

## Troubleshooting

### ModuleNotFoundError

If you get import errors, ensure you're in the correct virtual environment:

```bash
source .venv/bin/activate
pip install pitchpredict
```

### pybaseball connection issues

pybaseball fetches data from external sources. If you encounter connection issues:

1. Check your internet connection
2. Try again later (rate limiting may apply)
3. Enable caching to reduce repeated requests (enabled by default)

### Port already in use

If port 8056 is taken when starting the server:

```bash
pitchpredict serve --port 8080  # Use a different port
```
