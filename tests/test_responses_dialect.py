"""OpenAI Responses dialect: grouping, digests, compaction, proxy e2e."""

from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from cliffcompaction.cliff import compact, group_turns
from cliffcompaction.config import Config
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.dialects.openai_responses import DIALECT, digest_message
from cliffcompaction.engine import Engine
from cliffcompaction.proxy import create_app


def dev(text):
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def reasoning(i, summary_text=None):
    item = {"type": "reasoning", "id": f"rs_{i}", "encrypted_content": f"enc-{i}" * 10}
    if summary_text:
        item["summary"] = [{"type": "summary_text", "text": summary_text}]
    return item


def fcall(i, cmd="echo hi"):
    return {
        "type": "function_call",
        "id": f"fc_{i}",
        "call_id": f"call_{i}",
        "name": "exec_command",
        "arguments": json.dumps({"cmd": cmd}),
        "status": "completed",
    }


def fout(i, output="ok (exit 0)"):
    return {"type": "function_call_output", "call_id": f"call_{i}", "output": output}


def codex_session(n_steps, fat_every=2):
    """Codex-shaped history: head (developer + env + task), then n_steps of
    [reasoning, function_call, function_call_output]."""
    items = [
        dev("<skills_instructions>be a good agent</skills_instructions>"),
        user("<environment_context><cwd>/tmp</cwd></environment_context>"),
        user("Task: run the build. Codeword: PELICAN."),
    ]
    for i in range(n_steps):
        items.append(reasoning(i, summary_text=f"planning step {i}"))
        items.append(fcall(i, cmd=f"make step{i}"))
        fat = fat_every and i % fat_every == 0
        items.append(fout(i, output=("LOG " + "x" * 2000) if fat else f"step{i} ok"))
    return items


def r_body(items):
    return {
        "model": "gpt-5",
        "instructions": "You are a coding agent." + "x" * 200,
        "input": items,
        "store": False,
        "stream": True,
    }


# --- grouping ----------------------------------------------------------------


def test_model_step_stays_one_turn():
    items = codex_session(3)
    turns = group_turns(items, DIALECT)
    # leading head group + 3 step turns
    assert len(turns) == 4
    assert [i["type"] for i in turns[1]] == ["reasoning", "function_call", "function_call_output"]
    # partition, in order
    assert [i for t in turns for i in t] == items


def test_assistant_message_in_step_run():
    items = [user("task"), reasoning(0), assistant("thinking done"), fcall(0), fout(0)]
    turns = group_turns(items, DIALECT)
    assert len(turns) == 2
    assert len(turns[1]) == 4  # reasoning + message + call + output together


# --- digests -----------------------------------------------------------------


def test_digest_ignores_id_and_status():
    a = fcall(1)
    b = {**fcall(1), "id": "totally-different", "status": "in_progress"}
    assert digest_message(a) == digest_message(b)
    assert digest_message(fcall(1)) != digest_message(fcall(2))


def test_digest_reasoning_tracks_encrypted_content():
    assert digest_message(reasoning(1)) != digest_message(reasoning(2))
    assert digest_message(reasoning(1)) == digest_message(reasoning(1))


# --- compaction --------------------------------------------------------------


def test_compact_codex_history():
    items = codex_session(6)
    res = compact(items, DIALECT, Config(keep_recent=2))
    assert res is not None
    assert res.head_len == 3  # developer + env + task stay verbatim
    assert res.messages[:3] == items[:3]
    summary = res.messages[3]
    assert summary["role"] == "user"
    text = summary["content"][0]["text"]
    assert text.startswith(SUMMARY_HEADER)
    # content classes: signatures, small results kept, fat results dropped,
    # reasoning summaries folded as thinking, encrypted blobs never leak
    assert "[exec_command]" in text
    assert "step1 ok" in text
    assert "LOG " not in text
    assert "thinking: planning step 0" in text
    assert "enc-" not in text
    # the kept tail is the last 2 full steps, reasoning included
    kept = res.messages[4:]
    assert [i["type"] for i in kept[:3]] == ["reasoning", "function_call", "function_call_output"]
    assert kept[0]["encrypted_content"].startswith("enc-4")


def test_recompaction_drops_prior_summary():
    items = codex_session(6)
    res = compact(items, DIALECT, Config(keep_recent=2))
    grown = res.messages + codex_session(0)[:0]  # copy
    grown = res.messages + [reasoning(9), fcall(9), fout(9)]
    res2 = compact(grown, DIALECT, Config(keep_recent=2))
    assert res2 is not None
    text = res2.summary["content"][0]["text"]
    assert text.count(SUMMARY_HEADER) == 1
    assert res2.messages[3] == res2.summary
    assert res2.head_len == 3


# --- engine ------------------------------------------------------------------


def test_engine_prefix_substitution_on_input_key():
    cfg = Config(threshold_tokens=1, keep_recent=1)
    eng = Engine(cfg)
    items = codex_session(5)
    ctx = eng.prepare(r_body(items), DIALECT)
    assert ctx.compacted
    out = ctx.outgoing_body()
    assert len(out["input"]) < len(items)
    assert out["instructions"] == r_body(items)["instructions"]

    # follow-up: original history + one more step -> prefix match, same summary
    items2 = items + [reasoning(7), fcall(7), fout(7)]
    ctx2 = eng.prepare(r_body(items2), DIALECT)
    assert ctx2.modified
    out2 = ctx2.outgoing_body()
    assert out2["input"][3]["content"][0]["text"].startswith(SUMMARY_HEADER)
    assert out2["input"][-3:] == items2[-3:]


# --- proxy e2e ---------------------------------------------------------------


class Upstream:
    def __init__(self):
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "resp_1", "object": "response", "output": []})


def make(cfg: Config):
    up = Upstream()
    app = create_app(cfg, transport=httpx.MockTransport(up.handler))
    return TestClient(app), up


def test_proxy_responses_route_and_compaction():
    cfg = Config(
        threshold_tokens=2_000,
        keep_recent=1,
        anthropic_upstream="https://anthropic.example",
        openai_upstream="https://openai.example",
    )
    client, up = make(cfg)
    r = client.post("/v1/responses", json=r_body(codex_session(8)))
    assert r.status_code == 200
    req = up.requests[-1]
    assert req.url.host == "openai.example"
    sent = json.loads(req.content)
    assert sent["input"][3]["content"][0]["text"].startswith(SUMMARY_HEADER)
    assert sent["store"] is False  # rest of the body untouched


def test_proxy_responses_compact_endpoint_passthrough():
    cfg = Config(
        threshold_tokens=1,
        anthropic_upstream="https://anthropic.example",
        openai_upstream="https://openai.example",
    )
    client, up = make(cfg)
    raw = json.dumps(r_body(codex_session(4))).encode()
    r = client.post("/v1/responses/compact", content=raw)
    assert r.status_code == 200
    req = up.requests[-1]
    assert req.url.host == "openai.example"
    assert req.content == raw  # no dialect claims it: byte-verbatim


# --- Codex code mode (custom_tool_call `exec`) and kept results ---------------


def ccall(i, cmd):
    """Codex 0.163 code mode: a shell command wrapped in an `exec` script."""
    return {
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": f"cc_{i}",
        "name": "exec",
        "input": f"text(await tools.exec_command({{cmd:{json.dumps(cmd)}}}));\n",
    }


def cout(i, output, exit_code=0):
    payload = {"chunk_id": "c", "wall_time_seconds": 0.1, "exit_code": exit_code,
               "original_token_count": len(output) // 4, "output": output}
    return {
        "type": "custom_tool_call_output",
        "call_id": f"cc_{i}",
        "output": [
            {"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"},
            {"type": "input_text", "text": json.dumps(payload)},
        ],
    }


SKILL = "# Pick-place skill\n" + "".join(f"Rule {i}: check widget {i}.\n" for i in range(200))


def code_session(cmds):
    """Head, then one [reasoning, exec call, exec output] step per (cmd, output)."""
    items = [dev("<skills_instructions>x</skills_instructions>"),
             user("<environment_context><cwd>/w</cwd></environment_context>"),
             user("Task: pick the green ball.")]
    for i, (cmd, out) in enumerate(cmds):
        items += [reasoning(i), ccall(i, cmd), cout(i, out)]
    return items


def summary_of(res):
    return res.messages[res.head_len]["content"][0]["text"]


def test_code_mode_exec_summarized_by_command():
    items = code_session([("wc -l SKILL.md", "201 SKILL.md"), ("cat SKILL.md", SKILL), ("echo done", "done")])
    res = compact(items, DIALECT, Config(keep_recent=1))
    text = summary_of(res)
    assert "[exec_command] wc -l SKILL.md" in text
    assert "[exec_command] cat SKILL.md" in text
    assert "result: 201 SKILL.md" in text  # wrapper decoded, short output kept
    assert "Rule 7: check widget 7" not in text  # long output dropped by default
    assert "<kept-result" not in text  # feature off without a pattern


def test_keep_result_pattern_keeps_matching_reads_verbatim():
    log = "LOG " + "y" * 3000
    items = code_session([("cat SKILL.md", SKILL), ("make build", log), ("echo done", "done")])
    res = compact(items, DIALECT, Config(keep_recent=1, keep_result_pattern=r"SKILL\.md"))
    text = summary_of(res)
    assert '<kept-result call="[exec_command] cat SKILL.md">' in text
    assert SKILL.strip() in text  # verbatim, decoded from the exec wrapper
    assert "yyyy" not in text  # non-matching large result still dropped


def test_kept_results_survive_recompaction_and_rereads_replace():
    cfg = Config(keep_recent=1, keep_result_pattern=r"SKILL\.md")
    items = code_session([("cat SKILL.md", SKILL), ("make a", "a ok"), ("make b", "b ok")])
    first = compact(items, DIALECT, cfg)
    # The client resends the compacted history plus new steps; the old reads are gone.
    later = list(first.messages) + [reasoning(10), ccall(10, "make c"), cout(10, "c ok"),
                                    reasoning(11), ccall(11, "make d"), cout(11, "d ok")]
    second = compact(later, DIALECT, cfg)
    text = summary_of(second)
    assert text.count("<kept-result") == 1
    assert SKILL.strip() in text
    # A later read of the same call replaces the kept copy, in the same position.
    newer = SKILL.replace("Rule 7:", "Rule 7 (revised):")
    again = list(second.messages) + [reasoning(12), ccall(12, "cat SKILL.md"), cout(12, newer),
                                      reasoning(13), ccall(13, "make e"), cout(13, "e ok")]
    third = summary_of(compact(again, DIALECT, cfg))
    assert third.count("<kept-result") == 1
    assert "Rule 7 (revised):" in third



def test_anthropic_keeps_matching_tool_results():
    from cliffcompaction.dialects.anthropic import DIALECT as A
    msgs = [{"role": "user", "content": "Task: fix it."}]
    for i, (name, inp, out) in enumerate([("Read", {"file_path": "/w/SKILL.md"}, SKILL),
                                          ("Bash", {"command": "make"}, "z" * 3000),
                                          ("Bash", {"command": "echo ok"}, "ok")]):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": out}]})
    res = compact(msgs, A, Config(keep_recent=1, keep_result_pattern=r"SKILL\.md"))
    text = res.messages[res.head_len]["content"]
    assert SKILL.strip() in text and "zzzz" not in text
