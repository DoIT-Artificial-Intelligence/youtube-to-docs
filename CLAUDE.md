# YouTube to Docs

The repository root is a plugin for both Claude Code and Antigravity. Keep one copy of each component:

- `skills/youtube-to-docs/SKILL.md`: the only skill. How to call the `process_video` tool lives here, so update it when the tool's arguments change.
- `mcp_config.json`: the MCP server definition. Antigravity reads it from the plugin root, and `.claude-plugin/plugin.json` points Claude Code at it.
- `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json`: the Claude Code manifest and marketplace. The plugin has no `version`, so installs update with every commit.
- `plugin.json`: the Antigravity manifest. Its schema allows only `$schema`, `name` and `description`.

Check changes with `claude plugin validate .`.
