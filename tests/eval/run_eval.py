"""Đánh giá NLU + pipeline và xuất tests/report.md.

    python tests/eval/run_eval.py --mode fallback            # nhanh, không cần Ollama
    python tests/eval/run_eval.py --mode llm                 # Ollama thật (chậm: ~15-20 s/câu trên CPU)
    python tests/eval/run_eval.py --mode both --report       # cả hai + ghi tests/report.md
    python tests/eval/run_eval.py --mode fallback --set dev  # chỉ tập dev

Chỉ số:
- Độ chính xác intent: intent CHÍNH của NLU thuộc tập intent chấp nhận.
- Độ chính xác slot: mỗi slot kỳ vọng (giá trị trong state sau merge / slot bị xóa / hỏi lại / xác nhận).
- Tỷ lệ gọi tool đúng: kỳ vọng tool (tên + tham số) trong các kịch bản tests/conversations.
- Số ca bịa thông tin: menu mặc định, báo thành công khi chưa commit, sai tình trạng bàn, quán không có trong DB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
ROOT = os.path.dirname(TESTS)
sys.path[:0] = [ROOT, TESTS]
for s in (sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

import logging  # noqa: E402

logging.basicConfig(level=logging.ERROR)

from helpers import hallucination_checks, load_yaml, seed_context, subst  # noqa: E402
import yaml  # noqa: E402

TEMPLATE_DB = None


def fresh_db() -> None:
    """Mỗi ca dùng 1 bản sao DB sạch (ca xác nhận hủy/đặt có ghi DB)."""
    global TEMPLATE_DB
    from actions import db
    from core import entities

    if TEMPLATE_DB is None:
        d = tempfile.mkdtemp(prefix="eval_db_")
        TEMPLATE_DB = os.path.join(d, "template.db")
        os.environ["SQLITE_DB_PATH"] = TEMPLATE_DB
        db.init_database(reset=True)
        entities.reset_restaurant_cache()
    work = os.path.join(os.path.dirname(TEMPLATE_DB), "work.db")
    shutil.copy(TEMPLATE_DB, work)
    os.environ["SQLITE_DB_PATH"] = work


def make_agent(mode: str):
    os.environ["LLM_PROVIDER"] = "none" if mode == "fallback" else "ollama"
    from agent.agent import RestaurantAgent

    fresh_db()
    return RestaurantAgent(warmup=False)


def eval_case(agent, case: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    fresh_db()
    seed_context(agent, ctx)
    before = dict(agent.state.slots)
    t0 = time.time()
    try:
        reply = agent.ask(case["text"])
        crashed = None
    except Exception as e:  # pragma: no cover
        reply, crashed = "", repr(e)
    latency = time.time() - t0
    t = agent.last_trace if not crashed else {"intents": ["<crash>"], "blocks": [], "tools": []}
    res: Dict[str, Any] = {"text": case["text"], "latency": latency, "source": t.get("nlu_source"),
                           "llm_error": t.get("llm_error"), "intents_got": t.get("intents"),
                           "intents_exp": case["intents"], "reply": reply, "problems": []}
    res["intent_ok"] = bool(t.get("intents")) and t["intents"][0] in case["intents"]
    if not res["intent_ok"]:
        res["problems"].append(f"intent {t.get('intents')} ≠ {case['intents']}")
    slot_checks = []
    slots = agent.state.slots
    for k, v in (case.get("slots") or {}).items():
        ok = slots.get(k) == v
        slot_checks.append(ok)
        if not ok:
            res["problems"].append(f"slot {k}={slots.get(k)!r} ≠ {v!r}")
    for k in case.get("removed") or []:
        ok = before.get(k) is not None and slots.get(k) is None
        slot_checks.append(ok)
        if not ok:
            res["problems"].append(f"slot {k} chưa bị xóa (={slots.get(k)!r})")
    if case.get("clarify"):
        ok = bool(t.get("needs_clarification")) and "clarify" in t["blocks"]
        slot_checks.append(ok)
        if not ok:
            res["problems"].append("không hỏi lại câu mơ hồ")
    if case.get("confirmation"):
        ok = t.get("confirmation") == case["confirmation"] or (
            case["confirmation"] == "yes" and t["intents"][0] == "confirm") or (
            case["confirmation"] == "no" and t["intents"][0] == "deny")
        slot_checks.append(ok)
        if not ok:
            res["problems"].append(f"confirmation={t.get('confirmation')!r} ≠ {case['confirmation']!r}")
    res["slot_checks"] = slot_checks
    res["block_ok"] = True
    if case.get("blocks_any") and not set(case["blocks_any"]) & set(t["blocks"]):
        res["block_ok"] = False
        res["problems"].append(f"block {t['blocks']} thiếu {case['blocks_any']}")
    res["hallucinations"] = [] if crashed else hallucination_checks(agent, reply)
    if crashed:
        res["problems"].append(f"CRASH {crashed}")
    res["problems"] += res["hallucinations"]
    return res


def run_set(agent, name: str, path: str, contexts: Dict[str, Any], limit: int, verbose: bool) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)          # {tomorrow}... được thay ĐÚNG LÚC chạy từng ca (chạy qua nửa đêm)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n = 0
    for gname, g in data.items():
        for case in g["cases"]:
            if limit and n >= limit:
                break
            ctx = subst(contexts[case.get("context", g.get("context", "none"))] or {})
            r = eval_case(agent, subst(case), ctx)
            r["group"] = gname
            groups[gname].append(r)
            n += 1
            if verbose:
                mark = "✓" if not r["problems"] else "✗"
                print(f"  {mark} [{name}/{gname}] {case['text']}  -> {r['intents_got']} "
                      f"({r['source']}, {r['latency']:.1f}s) {'; '.join(r['problems'])}", flush=True)
    return groups


def run_scenarios(agent) -> Dict[str, Any]:
    from test_conversations import run_scenario

    out = []
    for path in sorted(glob.glob(os.path.join(TESTS, "conversations", "*.yaml"))):
        fresh_db()
        out.append(run_scenario(agent, path))
    return out


def summarize(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    cases = [c for g in groups.values() for c in g]
    slot_checks = [x for c in cases for x in c["slot_checks"]]
    lat = [c["latency"] for c in cases]
    return {
        "n": len(cases),
        "intent_acc": sum(c["intent_ok"] for c in cases) / max(1, len(cases)),
        "slot_acc": sum(slot_checks) / max(1, len(slot_checks)),
        "n_slot_checks": len(slot_checks),
        "case_pass": sum(not c["problems"] for c in cases) / max(1, len(cases)),
        "hallucinations": sum(len(c["hallucinations"]) for c in cases),
        "default_menu": sum(any(h.startswith("MENU_MẶC_ĐỊNH") for h in c["hallucinations"]) for c in cases),
        "false_success": sum(any(h.startswith("BÁO_THÀNH_CÔNG") for h in c["hallucinations"]) for c in cases),
        "llm_fail": sum(1 for c in cases if c["llm_error"] in ("LLM_UNAVAILABLE", "LLM_INVALID_OUTPUT")),
        "lat_avg": statistics.mean(lat) if lat else 0,
        "lat_p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else 0,
        "per_group": {g: (sum(c["intent_ok"] for c in cs), len(cs)) for g, cs in groups.items()},
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(results: Dict[str, Dict[str, Any]], path: str, meta: Dict[str, Any]) -> None:
    L = ["# Báo cáo kiểm thử chatbot đặt bàn", "",
         f"- Thời điểm chạy: {meta['when']}",
         f"- Model LLM: `{meta['model']}` (Ollama, CPU) · Embedding: `{meta['embedding']}`",
         "- Chế độ `llm` = Ollama + validator (dự phòng embedding nếu lỗi); `fallback` = chỉ NLU dự phòng.",
         "- Tập **held-out** không được dùng để chỉnh prompt/few-shot/câu mẫu/luật; có test tự động kiểm tra "
         "không trùng câu (`test_fewshot_and_examples_disjoint_from_test_sets`).", ""]
    L += ["## Tổng hợp", "",
          "| Chế độ | Tập | Số ca | Intent | Slot | Ca đạt hoàn toàn | Bịa thông tin | Menu mặc định | Báo thành công sai | LLM lỗi | Trễ TB / p95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, r in results.items():
        for set_name, s in r["sets"].items():
            L.append(f"| {mode} | {set_name} | {s['n']} | **{pct(s['intent_acc'])}** | {pct(s['slot_acc'])} "
                     f"({s['n_slot_checks']}) | {pct(s['case_pass'])} | {s['hallucinations']} | {s['default_menu']} "
                     f"| {s['false_success']} | {s['llm_fail']} | {s['lat_avg']:.1f}s / {s['lat_p95']:.1f}s |")
    L.append("")
    L += ["## Kịch bản nhiều lượt (tỷ lệ gọi tool đúng)", "",
          "| Chế độ | Kịch bản đạt | Lượt | Kỳ vọng tool đạt |", "|---|---|---|---|"]
    for mode, r in results.items():
        sc = r["scenarios"]
        te, tok = sum(x["tool_expect"] for x in sc), sum(x["tool_ok"] for x in sc)
        L.append(f"| {mode} | {sum(not x['failures'] for x in sc)}/{len(sc)} | {sum(x['turns'] for x in sc)} "
                 f"| {tok}/{te} ({pct(tok / max(1, te))}) |")
    L.append("")
    targets = []
    for mode, r in results.items():
        h = r["sets"].get("heldout")
        if h:
            targets.append(f"- [{mode}] Intent held-out ≥ 90%: **{'ĐẠT' if h['intent_acc'] >= 0.9 else 'CHƯA ĐẠT'}** "
                           f"({pct(h['intent_acc'])})")
        tot_fs = sum(s["false_success"] for s in r["sets"].values())
        tot_menu = sum(s["default_menu"] for s in r["sets"].values())
        targets.append(f"- [{mode}] 0 ca báo đặt bàn thành công khi chưa commit: **{'ĐẠT' if tot_fs == 0 else 'CHƯA ĐẠT'}** ({tot_fs})")
        targets.append(f"- [{mode}] 0 ca trả menu mặc định cho câu có nghĩa: **{'ĐẠT' if tot_menu == 0 else 'CHƯA ĐẠT'}** ({tot_menu})")
    L += ["## Mục tiêu", ""] + targets + [""]
    for mode, r in results.items():
        L += [f"## Độ chính xác intent theo nhóm — {mode}", "", "| Nhóm | " + " | ".join(r["sets"]) + " |",
              "|---|" + "---|" * len(r["sets"])]
        all_groups = sorted({g for s in r["sets"].values() for g in s["per_group"]})
        for g in all_groups:
            cells = []
            for s in r["sets"].values():
                ok, n = s["per_group"].get(g, (0, 0))
                cells.append(f"{ok}/{n}" if n else "—")
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")
    for mode, r in results.items():
        L += [f"## Danh sách ca sai — {mode}", ""]
        any_fail = False
        for set_name, groups in r["raw"].items():
            for g, cs in groups.items():
                for c in cs:
                    if c["problems"]:
                        any_fail = True
                        L.append(f"- **[{set_name}/{g}]** `{c['text']}` → intent {c['intents_got']} "
                                 f"(nguồn {c['source']}): " + "; ".join(c["problems"]))
        for x in r["scenarios"]:
            for f in x["failures"]:
                any_fail = True
                L.append("- **[kịch bản]** " + f.replace("\n", " "))
        if not any_fail:
            L.append("(không có)")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fallback", "llm", "both"], default="fallback")
    ap.add_argument("--set", choices=["dev", "heldout", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="ghi tests/report.md")
    ap.add_argument("--no-scenarios", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--json", help="lưu kết quả thô ra file JSON")
    args = ap.parse_args()

    import config

    with open(os.path.join(HERE, "contexts.yaml"), encoding="utf-8") as f:
        contexts = yaml.safe_load(f)
    sets = {"dev": os.path.join(HERE, "nlu_dev.yaml"), "heldout": os.path.join(HERE, "nlu_heldout.yaml")}
    if args.set != "all":
        sets = {args.set: sets[args.set]}
    modes = ["fallback", "llm"] if args.mode == "both" else [args.mode]
    results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        print(f"=== Chế độ {mode} ===", flush=True)
        agent = make_agent(mode)
        raw = {name: run_set(agent, name, p, contexts, args.limit, not args.quiet) for name, p in sets.items()}
        scen = [] if args.no_scenarios else run_scenarios(agent)
        results[mode] = {"raw": raw, "sets": {k: summarize(v) for k, v in raw.items()}, "scenarios": scen}
        for k, s in results[mode]["sets"].items():
            print(f"[{mode}/{k}] n={s['n']} intent={pct(s['intent_acc'])} slot={pct(s['slot_acc'])} "
                  f"pass={pct(s['case_pass'])} bịa={s['hallucinations']} menu={s['default_menu']} "
                  f"llm_lỗi={s['llm_fail']} trễ_TB={s['lat_avg']:.1f}s", flush=True)
        if scen:
            te, tok = sum(x["tool_expect"] for x in scen), sum(x["tool_ok"] for x in scen)
            print(f"[{mode}/kịch bản] đạt {sum(not x['failures'] for x in scen)}/{len(scen)}, tool {tok}/{te}")
            for x in scen:
                for f in x["failures"]:
                    print("   ✗", f.replace("\n", " | "))
    meta = {"when": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "model": config.ollama_model(),
            "embedding": config.embedding_model()}
    if args.report:
        write_report(results, os.path.join(TESTS, "report.md"), meta)
        print("Đã ghi", os.path.join(TESTS, "report.md"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
