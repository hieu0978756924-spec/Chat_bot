from core.normalizer import normalize_message
from core.schema import NLUOutput, SlotUpdate
from core.state import StateManager


def apply(sm, text, intents, updates=None, removals=None):
    nlu = NLUOutput(intents=intents, slot_updates={k: SlotUpdate(op=op, value=v) for k, (op, v) in (updates or {}).items()},
                    slot_removals=removals or [])
    return sm.apply(nlu, normalize_message(text))


def seeded():
    sm = StateManager()
    apply(sm, "x", ["search_restaurant"], {"cuisine": ("set", "Lẩu"), "date": ("set", "2026-11-18"),
                                           "time": ("set", "19:00"), "party_size": ("set", 4)})
    sm.set_results([{"ma_nha_hang": "NH001", "ten_nha_hang": "Lẩu Nhà Mình"},
                    {"ma_nha_hang": "NH004", "ten_nha_hang": "Góc Nướng Hàn"}])
    return sm


def test_change_main_criterion_keeps_date_time_party_and_clears_results():
    sm = seeded()
    sm.state.slots["restaurant_id"] = "NH001"
    r = apply(sm, "x", ["search_restaurant"], {"cuisine": ("set", "Chay"), "budget_max": ("set", 150000)})
    s = sm.state.slots
    assert (s["cuisine"], s["budget_max"]) == ("Chay", 150000)
    assert (s["date"], s["time"], s["party_size"]) == ("2026-11-18", "19:00", 4)
    assert s["restaurant_id"] is None and sm.state.last_search_results == []
    assert r.criteria_changed


def test_single_slot_change_keeps_everything_else():
    sm = seeded()
    r = apply(sm, "x", ["search_restaurant"], {"party_size": ("set", 6)})
    s = sm.state.slots
    assert s["party_size"] == 6 and s["cuisine"] == "Lẩu" and s["time"] == "19:00"
    assert len(sm.state.last_search_results) == 2 and not r.criteria_changed


def test_add_op_and_remove():
    sm = seeded()
    apply(sm, "x", ["search_restaurant"], {"party_size": ("add", 2)})
    assert sm.state.slots["party_size"] == 6
    apply(sm, "x", ["search_restaurant"], removals=["cuisine"])
    assert sm.state.slots["cuisine"] is None


def test_ordinal_reference_maps_to_last_results():
    sm = seeded()
    r = apply(sm, "đặt quán thứ 2", ["create_reservation"])
    assert r.reference == "resolved" and sm.state.slots["restaurant_id"] == "NH004"


def test_demonstrative_with_many_results_is_ambiguous():
    sm = seeded()
    assert apply(sm, "quán đó còn bàn không", ["check_availability"]).reference == "ambiguous"


def test_interrupt_pushes_task_stack():
    sm = seeded()
    sm.state.active_task = "search_restaurant"
    r = apply(sm, "x", ["policy_question"])
    assert r.interrupted_task == "search_restaurant" and sm.state.task_stack
