# Tests

This directory contains the test suite for `youtube-to-docs`.

## Running Tests with uv

To run the full test suite in parallel with all required dependencies, use:

```bash
uv run --all-extras --group test pytest -n auto
```

## Manual Tests

The following tests run outside of the standard test suite (they are skipped in CI) as they require manual setup and external credentials:

- `tests/test_workspace.py`: Verifies Google Drive/Workspace integration.
- `tests/test_sharepoint.py`: Verifies OneDrive/SharePoint integration.

### How to Run

To run these tests, you must have the required credentials configured and execute them using `uv` with all extras:

```bash
uv run --all-extras --group test python tests/test_workspace.py
uv run --all-extras --group test python tests/test_sharepoint.py
```
