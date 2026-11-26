from core.normalizer import normalize_message
from core.schema import NLUOutput, SlotUpdate


def run(agent, text, intents, updates=None, confirmation=None):
    nm = normalize_message(text)
    nlu = NLUOutput(intents=intents, confirmation=confirmation,
                    slot_updates={k: SlotUpdate(op="set", value=v) for k, v in (updates or {}).items()})
    merge = agent.sm.apply(nlu, nm)
    agent.tools.trace = []
    return agent.policy.decide(nlu, nm, merge, {})


def test_secondary_intent_never_overwrites_pending(fallback_agent):
    """Lỗi thật gặp khi chạy Ollama: [cancel_reservation, policy_question] -> pending hủy bị thay bằng phiếu hỗ trợ."""
    a = fallback_agent
    run(a, "hủy DB003 giúp mình, quy định về vũ trụ thế nào", ["cancel_reservation", "policy_question"],
        {"booking_code": "DB003"})
    assert a.state.pending_action["type"] == "cancel_reservation"
    run(a, "đồng ý", ["confirm"], confirmation="yes")
    assert [c["tool"] for c in a.tools.trace] == ["cancel_reservation"]


def test_confirmation_required_before_write(fallback_agent):
    a = fallback_agent
    blocks = run(a, "hủy DB003", ["cancel_reservation"], {"booking_code": "DB003"})
    assert [b["type"] for b in blocks] == ["confirm"]
    assert not any(c["tool"] == "cancel_reservation" for c in a.tools.trace)
    blocks = run(a, "không", ["deny"], confirmation="no")
    assert blocks[0]["type"] == "pending_cancelled" and a.state.pending_action is None


def test_no_default_menu_for_unknown(fallback_agent):
    blocks = run(fallback_agent, "abc xyz", ["out_of_scope"])
    assert blocks[0]["type"] == "out_of_scope"


def test_update_without_code_during_search_modifies_search(fallback_agent):
    a = fallback_agent
    run(a, "lẩu", ["search_restaurant"], {"cuisine": "Lẩu"})
    blocks = run(a, "đổi sang 6 người", ["update_reservation"], {"party_size": 6})
    assert blocks[0]["type"] == "restaurant_list" and a.state.slots["party_size"] == 6


def test_cancel_confirmation_with_other_content_asks_again(fallback_agent):
    """Lỗi thật khi chạy LLM: khiếu nại bị hiểu thành hủy, khách đáp 'ừ tạo phiếu giúp mình' -> KHÔNG được hủy."""
    a = fallback_agent
    run(a, "nhân viên nói không có đơn DB001", ["cancel_reservation"], {"booking_code": "DB001"})
    blocks = run(a, "ừ tạo phiếu giúp mình", ["confirm"], confirmation="yes")
    assert blocks[0]["type"] == "clarify" and "HỦY" in blocks[0]["text"]
    assert not any(c["tool"] == "cancel_reservation" for c in a.tools.trace)
    assert a.state.pending_action["type"] == "cancel_reservation"
    run(a, "có", ["confirm"], confirmation="yes")
    assert [c["tool"] for c in a.tools.trace] == ["cancel_reservation"]
