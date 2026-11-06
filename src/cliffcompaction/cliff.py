from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Config
from .dialects.base import SUMMARY_HEADER, Dialect


@dataclass
class CompactResult:
    messages: list[dict]  # the compacted message list
    head_len: int  # number of verbatim head messages
    summary: dict  # the synthetic summary message
    cut: int  # index into the INPUT list: messages[cut:] were kept verbatim


KEPT_HEADER = "Tool results kept verbatim from earlier reads:"
_KEPT_RE = re.compile(r'<kept-result call=("(?:[^"\\]|\\.)*")>\n(.*?)\n</kept-result>', re.DOTALL)


def _kept_results(to_compact: list[list[dict]], dialect: Dialect, cfg: Config) -> str:
    """Verbatim results of calls matching cfg.keep_result_pattern, as a summary section.

    Results kept by an earlier summary are carried forward (re-compaction
    drops that summary), and a later read of the same call replaces an
    earlier one. Order is first-read order, so the section stays stable
    across compactions. Deterministic in the input, like the rest of compact().
    """
    if not (cfg.keep_result_pattern and dialect.tool_calls and dialect.tool_results):
        return ""
    pattern = re.compile(cfg.keep_result_pattern)
    kept: dict[str, str] = {}
    calls: dict[str, str] = {}
    for turn in to_compact:
        for msg in turn:
            if dialect.is_summary_message(msg):
                if dialect.message_text is not None:
                    for m in _KEPT_RE.finditer(dialect.message_text(msg)):
                        kept[json.loads(m.group(1))] = m.group(2)
                continue
            for call_id, text in dialect.tool_calls(msg):
                calls[call_id] = text
            for call_id, out in dialect.tool_results(msg):
                call = calls.get(call_id)
                # Short results already appear in the summary as `result:` lines.
                if call and pattern.search(call) and len(out) > cfg.result_max_chars:
                    kept[call] = out
    blocks = [f"<kept-result call={json.dumps(call)}>\n{out}\n</kept-result>"
              for call, out in kept.items()]
    return KEPT_HEADER + "\n\n" + "\n\n".join(blocks) if blocks else ""


def group_turns(body: list[dict], dialect: Dialect) -> list[list[dict]]:
    """Group messages into assistant-step turns.

    A turn starts at each assistant message and includes the following
    non-assistant messages (its observations). Leading non-assistant
    messages (e.g. a prior summary) form their own group so they can be
    dropped cleanly on re-compaction.

    Dialects may override the boundary rule (dialect.group_turns) while
    keeping the same contract: a partition of the input, in order, with
    leading non-assistant messages in their own group.
    """
    if dialect.group_turns is not None:
        return dialect.group_turns(body)
    turns: list[list[dict]] = []
    current: list[dict] | None = None
    for msg in body:
        if dialect.is_assistant(msg):
            if current is not None:
                turns.append(current)
            current = [msg]
        elif current is None:
            current = [msg]
        else:
            current.append(msg)
    if current is not None:
        turns.append(current)
    return turns


def compact(messages: list[dict], dialect: Dialect, cfg: Config) -> CompactResult | None:
    """Compact a message list. Returns None if there is nothing to gain.

    Does not mutate the input; kept messages are passed by reference,
    never rebuilt.
    """
    # Head: everything before the first assistant message...
    first_assistant = next(
        (i for i, m in enumerate(messages) if dialect.is_assistant(m)), None
    )
    if first_assistant is None:
        return None
    head_len = first_assistant
    # ...minus trailing summary messages (re-compaction must not absorb them)
    # and messages that may not precede the injected user summary.
    while head_len > 0 and (
        dialect.is_summary_message(messages[head_len - 1])
        or (
            dialect.trim_from_head is not None
            and dialect.trim_from_head(messages[head_len - 1])
        )
    ):
        head_len -= 1

    body = messages[head_len:]
    turns = group_turns(body, dialect)

    keep_recent = max(0, cfg.keep_recent)
    if len(turns) <= keep_recent:
        return None
    to_compact = turns[: len(turns) - keep_recent]
    to_keep = turns[len(turns) - keep_recent :]

    parts: list[str] = []
    kept_section = _kept_results(to_compact, dialect, cfg)
    if kept_section:
        parts.append(kept_section)
    for turn in to_compact:
        for msg in turn:
            parts.extend(dialect.summarize_message(msg, cfg))

    summary_text = (
        SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(parts) if parts else SUMMARY_HEADER
    )
    summary = dialect.user_message(summary_text)

    kept: list[dict] = []
    for turn in to_keep:
        kept.extend(turn)

    new_messages = [*messages[:head_len], summary, *kept]
    if len(new_messages) >= len(messages):
        return None  # no reduction — treat as nothing to compact

    cut = len(messages) - len(kept)
    return CompactResult(messages=new_messages, head_len=head_len, summary=summary, cut=cut)
