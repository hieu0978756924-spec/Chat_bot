"""Kịch bản hội thoại nhiều lượt (tests/conversations/*.yaml).

Mỗi lượt kiểm tra: intent, slot trong state, tool + tham số đã gọi, block phản hồi,
và kiểm tra chống bịa (helpers.hallucination_checks).
Chạy 2 chế độ: NLU dự phòng (luôn chạy) và Ollama thật (RUN_OLLAMA_TESTS=1).
"""

import glob
import os

import pytest

from helpers import HERE, hallucination_checks, load_yaml

SCENARIOS = sorted(glob.glob(os.path.join(HERE, "conversations", "*.yaml")))


def check_turn(agent, reply, exp):
    """Trả về (danh sách lỗi, số kỳ vọng tool, số kỳ vọng tool đạt)."""
    t = agent.last_trace
    errors = []
    slots = agent.state.slots
    if "intents_any" in exp and not set(exp["intents_any"]) & set(t["intents"]):
        errors.append(f"intent {t['intents']} ∉ {exp['intents_any']}")
    for k, v in exp.get("slots", {}).items():
        if slots.get(k) != v:
            errors.append(f"slot {k}={slots.get(k)!r} (mong đợi {v!r})")
    for k, v in exp.get("slot_max", {}).items():
        if slots.get(k) is None or slots[k] > v:
            errors.append(f"slot {k}={slots.get(k)!r} (mong đợi ≤ {v})")
    tool_ok = 0
    for want in exp.get("tools", []):
        calls = [c for c in t["tools"] if c["tool"] == want["tool"]]
        if any(all(c["params"].get(p) == pv for p, pv in want.get("params", {}).items()) for c in calls):
            tool_ok += 1
        else:
            errors.append(f"thiếu tool {want['tool']}({want.get('params')}); đã gọi: "
                          f"{[(c['tool'], c['params']) for c in t['tools']]}")
    for tname in exp.get("tools_not", []):
        if any(c["tool"] == tname for c in t["tools"]):
            errors.append(f"KHÔNG được gọi tool {tname}")
    for b in exp.get("blocks_all", []):
        if b not in t["blocks"]:
            errors.append(f"thiếu block {b}; có {t['blocks']}")
    if exp.get("blocks_any") and not set(exp["blocks_any"]) & set(t["blocks"]):
        errors.append(f"cần 1 trong block {exp['blocks_any']}; có {t['blocks']}")
    for s in exp.get("reply_contains", []):
        if s not in reply:
            errors.append(f"phản hồi thiếu '{s}'")
    for s in exp.get("reply_not_contains", []):
        if s in reply:
            errors.append(f"phản hồi KHÔNG được chứa '{s}'")
    errors += hallucination_checks(agent, reply)
    return errors, len(exp.get("tools", [])), tool_ok


def run_scenario(agent, path):
    sc = load_yaml(path)
    agent.reset()
    failures, n_tool, n_tool_ok, n_turns = [], 0, 0, 0
    for i, turn in enumerate(sc["turns"], 1):
        reply = agent.ask(turn["user"])
        errs, nt, nok = check_turn(agent, reply, turn.get("expect", {}))
        n_tool, n_tool_ok, n_turns = n_tool + nt, n_tool_ok + nok, n_turns + 1
        if errs:
            failures.append(f"[{sc['name']} lượt {i}] {turn['user']!r}\n    " + "\n    ".join(errs)
                            + f"\n    trace: intents={agent.last_trace['intents']} "
                              f"src={agent.last_trace['nlu_source']} slots={agent.state.filled_slots()}")
    return {"name": sc["name"], "failures": failures, "tool_expect": n_tool, "tool_ok": n_tool_ok,
            "turns": n_turns}


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: os.path.basename(p))
def test_scenario_fallback_nlu(fallback_agent, path):
    r = run_scenario(fallback_agent, path)
    assert not r["failures"], "\n".join(r["failures"])


@pytest.mark.ollama
@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: os.path.basename(p))
def test_scenario_ollama(llm_agent, path):
    r = run_scenario(llm_agent, path)
    assert not r["failures"], "\n".join(r["failures"])
