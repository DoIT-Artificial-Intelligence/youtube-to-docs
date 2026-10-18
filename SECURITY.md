# Security Policy

## Supported Versions

youtube-to-docs is released often and only the latest release on
[PyPI](https://pypi.org/project/youtube-to-docs/) gets security fixes. Fixes
ship as a new release, not as backports to older versions.

| Version          | Supported          |
| ---------------- | ------------------ |
| Latest release   | :white_check_mark: |
| Anything older   | :x:                |

To upgrade:

```bash
uv tool upgrade youtube-to-docs      # if installed with `uv tool install`
uvx youtube-to-docs@latest --help    # if run with uvx
```

## Reporting a Vulnerability

**Please do not open a public issue, pull request, or discussion for a
security problem.**

Report it privately through GitHub:
[Report a vulnerability](https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**).

Please include:

- The version (`youtube-to-docs --version`) and how you run it (CLI, web app,
  MCP server, or plugin)
- What an attacker could do and what they need first (for example, network
  access to the web app or a crafted video)
- Steps or a proof of concept to reproduce it
- Any fix you suggest

Do not include real API keys, OAuth tokens, or other people's data in the
report. Remove or replace them first.

### What to expect

- **Acknowledgement** within 3 business days.
- **First assessment** (whether we accept the report, and how severe it is)
  within 10 business days.
- **Progress updates** at least every 2 weeks until it is resolved.
- **If we accept it**, we fix it in a new release, publish a GitHub security
  advisory, and request a CVE when it's warranted. We'll credit you in the
  advisory unless you ask us not to.
- **If we decline it**, we'll tell you why. You are welcome to disagree and
  send more details.

Please give us a reasonable time to release a fix before you disclose the
problem publicly. We'll agree on a disclosure date with you.

## Scope

**In scope:** the code in this repository. This includes the
`youtube-to-docs` CLI, the `youtube-to-docs-app` web app, the MCP server, the
Claude Code / Antigravity plugin and skill, and the GitHub Actions workflows.

**Out of scope:**

- Problems in third-party services (YouTube, Google Gemini or Vertex AI,
  AWS Bedrock, Azure AI Foundry, Hugging Face, Google Drive, Microsoft 365).
  Report those to the vendor.
- Problems in a dependency that we don't make worse. Report those upstream.
  We still want to hear if we pin a vulnerable version.
- Findings that need an attacker who already controls your machine, your
  user account, or your credential files.
- Running the web app on a network without authentication in front of it.
  This is documented behavior (see below).
- Wrong or harmful text that an AI model produces in a summary, unless our
  code makes it possible to exploit (for example, a prompt injection that
  leaks credentials or writes files outside the output folder).

## Using youtube-to-docs Safely

### Credentials

The tool reads API keys from environment variables, such as `GEMINI_API_KEY`,
`YOUTUBE_DATA_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, `AZURE_FOUNDRY_API_KEY`
and `HF_TOKEN`. It stores OAuth files in your home directory:

| File                            | Used for                           |
| ------------------------------- | ---------------------------------- |
| `~/.google_client_secret.json`  | Google Drive OAuth client          |
| `~/.google_client_token.json`   | Google Drive access/refresh token  |
| `~/.azure_client.json`          | Microsoft 365 app registration     |
| `~/.msal_token_cache.json`      | Microsoft 365 access/refresh token |

- Never commit these files or keys. Keep them readable only by you
  (for example, `chmod 600 ~/.google_client_token.json`).
- Use keys with the least access they need, and restrict them to the APIs
  this tool uses.
- If a key or token leaks, revoke it with the provider right away. To force
  a new OAuth sign-in, delete the token file.

### Web app

`youtube-to-docs-app` has **no authentication**. By default it only listens
on `127.0.0.1`. If you pass `--host 0.0.0.0` or another address, anyone who
can reach the port can start jobs that spend your API credits and can read the
generated files. Only do this on a network you trust, or put an authenticating
reverse proxy in front of it.

### Generated output

Transcripts, summaries and other output come from untrusted videos and from AI
models. Treat them as untrusted content. Review them before you publish or
share them, especially HTML files.

## How We Keep the Project Secure

- [Dependabot](.github/dependabot.yml) watches Python dependencies and GitHub
  Actions for known vulnerabilities.
- `uv.lock` pins every dependency. `[tool.uv] constraint-dependencies` in
  `pyproject.toml` moves vulnerable transitive dependencies up to patched
  versions.
- A CycloneDX software bill of materials is kept at
  [`sbom.cdx.json`](sbom.cdx.json).
- The CI and PyPI publish workflows run with read-only `contents` permission.
  Only the Dependabot auto-merge workflow has write access.
