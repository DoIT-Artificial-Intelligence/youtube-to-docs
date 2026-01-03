# Development Guide

## Prerequisites

- Python 3.14 or higher
- `uv`: A fast Python package and project manager.

### Installing `uv`

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more information, see the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

To run the script locally using the entry point defined in `pyproject.toml`:

```bash
uv run youtube-to-docs --model gemini-3-flash-preview
```

Alternatively, you can run it as a module:

```bash
uv run python -m youtube_to_docs.main --model gemini-3-flash-preview
```

## Running Tests

We use `pytest` for testing.

### Using `uv` (Recommended)

To run tests with all dependencies automatically handled:

```bash
uv run --group test pytest
```

## Utilities

### Cleanup

To delete all generated artifacts:

**Bash:**
```bash
rm -rf youtube-to-docs-artifacts/
```

**PowerShell:**
```powershell
Remove-Item -Path "youtube-to-docs-artifacts" -Recurse -Force -ErrorAction SilentlyContinue
```

## Project Structure

- `main.py`: Main application script.
- `tests/`: Directory containing test files.
- `pyproject.toml`: Project configuration and dependencies.

## Tooling

This project uses modern Python tooling for code quality:

- **Ruff**: For linting and code formatting.
- **Ty**: For static type checking.

These tools are configured in `pyproject.toml`.

### Running Ruff

To check for linting errors:

```bash
uv tool run ruff check .
```

To fix fixable linting errors automatically:

```bash
uv tool run ruff check --fix .
```

To format the code:

```bash
uv tool run ruff format .
```

### Running Ty (Type Checking)

To run type checks:

```bash
uv tool run ty check
```

### Pre-commit

To run pre-commit hook:

```bash
uv tool run pre-commit run --all-files
```

## Documentation

We use `MkDocs` with the `Material` theme for documentation.

### Build and Serve Locally

To preview the documentation locally:

```bash
uv run mkdocs serve
```

This will start a local server, usually at `http://127.0.0.1:8000`.

To build the static site (output will be in the `site/` directory):

```bash
uv run mkdocs build
```

### Deployment

The documentation is automatically built and deployed to GitHub Pages on every push to the `main` branch via GitHub Actions.

If you need to deploy manually:

```bash
uv run mkdocs gh-deploy
```

## Release to PyPI

To publish a new version of the package to PyPI, follow these steps:

1.  **Build the package**:
    This will create a `dist/` directory with the distribution files.
    ```bash
    uv tool run --from build pyproject-build
    ```

2.  **Upload to PyPI**:
    Use `twine` to upload the distribution files. 
    
    If your `.pypirc` is already configured with your API key:
    ```bash
    uv tool run twine upload dist/*
    ```

### Quick Deploy (One-liner)

To build and deploy in one command (requires `.pypirc` configuration):
```powershell
Remove-Item -Recurse -Force dist; uv tool run --from build pyproject-build; uv tool run twine upload dist/*
```

```bash
rm -rf dist; uv tool run --from build pyproject-build; uv tool run twine upload dist/*
```

## Continuous Deployment

A manual GitHub Action is available to automate this process. 

1. Ensure you have a `PYPI_API_TOKEN` secret configured in your repository settings.
2. Go to the "Actions" tab in your repository.
3. Select the "PyPI Publish" workflow.
4. Click "Run workflow".
