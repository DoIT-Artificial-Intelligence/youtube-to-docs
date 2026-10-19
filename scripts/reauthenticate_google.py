"""
Script to re-authenticate with Google services by deleting the cached token
and triggering a new OAuth flow.

Usage:
    uv sync --extra workspace && uv run scripts/reauthenticate_google.py
"""

from pathlib import Path


def reauthenticate_google() -> bool:
    """
    Deletes the cached Google OAuth token and triggers a new authentication flow.
    Returns True if successful, False otherwise.
    """
    token_file = Path.home() / ".google_client_token.json"
    creds_file = Path.home() / ".google_client_secret.json"

    if token_file.exists():
        try:
            token_file.unlink()
            print(f"Deleted cached token: {token_file}")
        except Exception as e:
            print(f"Error deleting token: {e}")
            return False
    else:
        print("No cached token found. Starting fresh authentication.")

    if not creds_file.exists():
        print(f"Error: Client secrets not found at {creds_file}")
        print("Please ensure you have your Google OAuth client secrets JSON file.")
        return False

    try:
        # Lazy import to avoid dependency issues for non-workspace users
        from youtube_to_docs.storage import get_google_credentials

        # With the token gone, this opens a browser sign-in and saves a new one.
        get_google_credentials()
        print(f"Successfully re-authenticated. Token saved to {token_file}")
        return True
    except ImportError:
        print(
            "Error: google-auth-oauthlib not installed. "
            "Run 'uv sync --extra workspace' to install it."
        )
        return False
    except Exception as e:
        print(f"Error during re-authentication: {e}")
        return False


if __name__ == "__main__":
    reauthenticate_google()
