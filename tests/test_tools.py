import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_server import tools


@pytest.fixture(autouse=True)
def reset_state():
    """Every test gets a clean slate -- write tools mutate shared state
    (LISTING_DB, PENDING_ACTIONS), so tests must not leak into each other."""
    tools.reset_state()
    yield
    tools.reset_state()


# ---------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------

def test_lookup_vin_found():
    result = tools.lookup_vin("1FTFW1ET5DFC10312")
    assert result["make"] == "Ford"
    assert result["model"] == "F-150"


def test_lookup_vin_not_found_raises():
    with pytest.raises(tools.ToolError, match="No vehicle found"):
        tools.lookup_vin("NOT-A-REAL-VIN")


def test_get_market_price_applies_depreciation():
    newer = tools.get_market_price("Honda", "Accord", 2023, 8100)
    older = tools.get_market_price("Honda", "Accord", 2015, 112400)
    assert newer["estimated_market_price"] > older["estimated_market_price"]


def test_get_market_price_unknown_make_model_raises():
    with pytest.raises(tools.ToolError, match="No market comp data"):
        tools.get_market_price("Yugo", "GV", 1988, 200000)


def test_get_market_price_never_goes_below_floor():
    # Extremely old/high-mileage car shouldn't produce a negative or
    # unrealistically tiny price
    result = tools.get_market_price("Chrysler", "Town & Country", 1990, 500000)
    assert result["estimated_market_price"] >= 1000


def test_check_inventory_returns_all_active_by_default():
    listings = tools.check_inventory()
    assert len(listings) == 5
    assert all(l["status"] == "active" for l in listings)


def test_check_inventory_filters_by_dealer():
    listings = tools.check_inventory(dealer_id="D-01")
    assert len(listings) == 3
    assert all(l["dealer_id"] == "D-01" for l in listings)


def test_check_compliance_flags_stale_listing():
    # L-1003 has days_on_lot=41, L-1005 has 67 -- only 1005 exceeds the 45-day threshold
    result = tools.check_compliance("L-1005")
    assert result["compliant"] is False
    assert any(i["rule"] == "max_days_on_lot" for i in result["issues"])


def test_check_compliance_passes_healthy_listing():
    result = tools.check_compliance("L-1002")  # 5 days on lot, reasonably priced
    assert result["compliant"] is True
    assert result["issues"] == []


def test_check_compliance_unknown_listing_raises():
    with pytest.raises(tools.ToolError, match="No listing found"):
        tools.check_compliance("L-9999")


# ---------------------------------------------------------------------
# Write tools -- must NOT mutate state directly
# ---------------------------------------------------------------------

def test_update_listing_price_does_not_mutate_immediately():
    original_price = tools.LISTING_DB["L-1001"]["price"]
    action = tools.update_listing_price("L-1001", 29999, requested_by="pricing_agent")

    assert action["status"] == "pending"
    assert tools.LISTING_DB["L-1001"]["price"] == original_price  # unchanged!
    assert action["payload"]["new_price"] == 29999


def test_update_listing_price_invalid_listing_raises():
    with pytest.raises(tools.ToolError, match="No listing found"):
        tools.update_listing_price("L-9999", 20000)


def test_update_listing_price_rejects_non_positive_price():
    with pytest.raises(tools.ToolError, match="Invalid price"):
        tools.update_listing_price("L-1001", -500)


def test_flag_compliance_issue_does_not_mutate_immediately():
    action = tools.flag_compliance_issue("L-1005", "Stale listing", severity="warning")
    assert action["status"] == "pending"
    assert tools.LISTING_DB["L-1005"]["status"] == "active"  # unchanged!


def test_flag_compliance_issue_rejects_invalid_severity():
    with pytest.raises(tools.ToolError, match="Invalid severity"):
        tools.flag_compliance_issue("L-1005", "reason", severity="urgent")


# ---------------------------------------------------------------------
# Human-in-the-loop approval flow
# ---------------------------------------------------------------------

def test_approve_action_executes_price_update():
    action = tools.update_listing_price("L-1001", 29999)
    result = tools.approve_action(action["action_id"])

    assert result["status"] == "approved"
    assert tools.LISTING_DB["L-1001"]["price"] == 29999  # NOW it's changed


def test_approve_action_executes_compliance_flag():
    action = tools.flag_compliance_issue("L-1005", "Stale listing")
    tools.approve_action(action["action_id"])

    assert tools.LISTING_DB["L-1005"]["status"] == "flagged"


def test_reject_action_does_not_mutate():
    action = tools.update_listing_price("L-1001", 29999)
    original_price = tools.LISTING_DB["L-1001"]["price"]
    result = tools.reject_action(action["action_id"])

    assert result["status"] == "rejected"
    assert tools.LISTING_DB["L-1001"]["price"] == original_price


def test_cannot_approve_twice():
    action = tools.update_listing_price("L-1001", 29999)
    tools.approve_action(action["action_id"])
    with pytest.raises(tools.ToolError, match="already approved"):
        tools.approve_action(action["action_id"])


def test_cannot_approve_unknown_action():
    with pytest.raises(tools.ToolError, match="No pending action"):
        tools.approve_action("nonexistent")


def test_list_pending_actions_excludes_resolved():
    a1 = tools.update_listing_price("L-1001", 29999)
    a2 = tools.flag_compliance_issue("L-1005", "reason")
    tools.approve_action(a1["action_id"])  # resolve one

    pending = tools.list_pending_actions()
    pending_ids = [p["action_id"] for p in pending]
    assert a1["action_id"] not in pending_ids
    assert a2["action_id"] in pending_ids


def test_reset_state_restores_original_prices():
    tools.update_listing_price("L-1001", 1)
    action_id = list(tools.PENDING_ACTIONS.keys())[0]
    tools.approve_action(action_id)
    assert tools.LISTING_DB["L-1001"]["price"] == 1

    tools.reset_state()
    assert tools.LISTING_DB["L-1001"]["price"] == 31900  # back to original
    assert tools.list_pending_actions() == []
