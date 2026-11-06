# Changelog

This fork (`BingqingChen/cliffcompaction`) tracks
[`nguyenvuthientrang/cliffcompaction`](https://github.com/nguyenvuthientrang/cliffcompaction), the
`upstream` remote. This file lists only the fork's own changes. For upstream history, read upstream's
git log.

Every change the fork makes gets an entry here, in the same commit as the change. The entry names the
behaviour, the reason, and the new options. Each sync with upstream records its upstream commit
below.

**Upstream base:** `b48d660` ("Banner: absolute URL for PyPI"), upstream `main` as of 2026-11-06.
Upstream's version is `0.1.0`; the fork has not changed it.

## Unreleased

### Added

- **`--keep-result-pattern` / `CLIFF_KEEP_RESULT_PATTERN`** (`ab4609e`): a regex matched against a
  tool call's text. The results of matching calls survive compaction:
  - They are kept verbatim in the summary at any size, and carried into every later summary.
  - So reference files an agent has read (skills, READMEs) don't need re-reading after compaction.
  - A later read of the same call replaces the kept copy; order stays first-read.
  - Off by default, and dropped at escalation rung 2. `/__cliff__/status` reports it.
  - Dialects expose `tool_calls` / `tool_results` / `message_text` hooks for this: Responses,
    Anthropic, Chat Completions.

### Fixed

- **Codex `exec` calls are summarized with their command** (`ab4609e`):
  - Codex 0.163+ sends shell commands as `custom_tool_call` items named `exec`, with the command in
    `input`. The Responses dialect read only `arguments`, so every such call was summarized as a bare
    `[exec] `.
  - Summaries now show the command.
  - `custom_tool_call_output` results are decoded and treated like `function_call_output`.
- **Multi-call code-mode exec scripts are summarized whole** (`3acc54a`):
  - A script used to be labelled by its first `tools.<name>(` call only. So a script that read
    `TASK.txt` and two `SKILL.md` files was keyed as `cat TASK.txt`, and the skills fell out at the
    first compaction despite `--keep-result-pattern`.
  - Scripts with more than one tool call now keep their full text.
