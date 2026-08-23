"""
mcp_server/tools.py
---------------------
The actual tool logic, kept as plain Python functions with no MCP or LLM
dependency. mcp_server/server.py wraps these with @mcp.tool() decorators.
Keeping logic here means every tool is unit-testable without spinning up
an MCP server or making any network/LLM call.

Read tools (lookup_vin, get_market_price, check_inventory, check_compliance)
execute immediately and return data.

Write tools (update_listing_price, flag_compliance_issue) do NOT mutate
state directly. They create a PendingAction and return it for approval --
approve_action() or reject_action() is what actually executes or discards
it. This is the human-in-the-loop guardrail: an agent can *propose* a
write, but only a human approval can make it real.

State persistence: LISTING_DB and PENDING_ACTIONS are saved to
data/state.json after every mutation and reloaded on import. This matters
because the MCP server runs as a fresh subprocess per CLI invocation (see
run_agent.py) -- without persistence, a price change proposed during
`python run_agent.py "some query"` would vanish before a later
`python run_agent.py --approve <id>` call could see it, since that's a
completely separate process with its own in-memory state otherwise.
"""

import sys
import os
import uuid
import copy
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.dealership_data import (
    VEHICLE_DB, LISTING_DB, MARKET_COMP_TABLE,
    MAX_DAYS_ON_LOT_WARNING, MIN_PRICE_FLOOR_FRACTION_OF_MARKET,
)

_ORIGINAL_LISTING_DB = copy.deepcopy(LISTING_DB)
_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "state.json")


class ToolError(Exception):
    """Raised for tool-level errors (not found, invalid input) -- kept
    distinct from unexpected exceptions so callers can handle them as
    normal tool output rather than a crash."""
    pass


@dataclass
class PendingAction:
    action_id: str
    action_type: str            # "update_listing_price" | "flag_compliance_issue"
    payload: dict
    status: str = "pending"     # "pending" | "approved" | "rejected"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None


PENDING_ACTIONS: dict[str, PendingAction] = {}


# ---------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------

def _save_state():
    state = {
        "listings": LISTING_DB,
        "pending_actions": {k: asdict(v) for k, v in PENDING_ACTIONS.items()},
    }
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_state():
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return  # corrupt or unreadable state file -- fall back to defaults rather than crash

    LISTING_DB.clear()
    LISTING_DB.update(state.get("listings", {}))
    PENDING_ACTIONS.clear()
    for action_id, payload in state.get("pending_actions", {}).items():
        PENDING_ACTIONS[action_id] = PendingAction(**payload)


def reset_state():
    """Restores LISTING_DB and PENDING_ACTIONS to their initial state and
    clears the persisted state file."""
    LISTING_DB.clear()
    LISTING_DB.update(copy.deepcopy(_ORIGINAL_LISTING_DB))
    PENDING_ACTIONS.clear()
    if os.path.exists(_STATE_FILE):
        os.remove(_STATE_FILE)


_load_state()  # pick up any state persisted by a previous process


# ---------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------

def lookup_vin(vin: str) -> dict:
    """Looks up a vehicle by VIN. Raises ToolError if not found."""
    vehicle = VEHICLE_DB.get(vin)
    if vehicle is None:
        raise ToolError(f"No vehicle found for VIN '{vin}'.")
    return dict(vehicle)


def get_market_price(make: str, model: str, year: int, mileage: int) -> dict:
    """Estimates market value using a simplified depreciation model.
    Raises ToolError if there's no comp data for this make/model."""
    key = (make, model)
    comp = MARKET_COMP_TABLE.get(key)
    if comp is None:
        raise ToolError(f"No market comp data available for {make} {model}.")

    current_year = datetime.now().year
    age_years = max(0, current_year - year)
    depreciation = age_years * comp["depreciation_per_year"] + (mileage / 1000) * comp["depreciation_per_1k_miles"]
    estimated_price = max(1000, round(comp["base_price"] - depreciation, -2))  # floor at $1000, round to nearest $100

    return {
        "make": make, "model": model, "year": year, "mileage": mileage,
        "estimated_market_price": estimated_price,
        "basis": {
            "base_price": comp["base_price"],
            "age_years": age_years,
            "depreciation_applied": round(depreciation, 2),
        },
    }


def check_inventory(dealer_id: Optional[str] = None) -> list[dict]:
    """Returns active listings, optionally filtered by dealer_id."""
    listings = [dict(l) for l in LISTING_DB.values() if l["status"] == "active"]
    if dealer_id:
        listings = [l for l in listings if l["dealer_id"] == dealer_id]
    return listings


def check_compliance(listing_id: str) -> dict:
    """Evaluates a listing against compliance rules and returns findings.
    Read-only -- does not itself flag anything (that's flag_compliance_issue,
    a write action requiring approval)."""
    listing = LISTING_DB.get(listing_id)
    if listing is None:
        raise ToolError(f"No listing found for '{listing_id}'.")

    vehicle = VEHICLE_DB.get(listing["vin"])
    if vehicle is None:
        raise ToolError(f"Listing '{listing_id}' references unknown VIN '{listing['vin']}'.")

    issues = []

    if listing["days_on_lot"] > MAX_DAYS_ON_LOT_WARNING:
        issues.append({
            "rule": "max_days_on_lot",
            "detail": f"On lot {listing['days_on_lot']} days, exceeds the {MAX_DAYS_ON_LOT_WARNING}-day warning threshold.",
        })

    try:
        market = get_market_price(vehicle["make"], vehicle["model"], vehicle["year"], vehicle["mileage"])
        price_floor = market["estimated_market_price"] * MIN_PRICE_FLOOR_FRACTION_OF_MARKET
        if listing["price"] < price_floor:
            issues.append({
                "rule": "price_below_floor",
                "detail": f"Listed at ${listing['price']:,}, which is below "
                          f"{MIN_PRICE_FLOOR_FRACTION_OF_MARKET:.0%} of estimated market value "
                          f"(${market['estimated_market_price']:,}) -- possible data-entry error.",
            })
    except ToolError:
        pass  # no comp data for this make/model -- can't check the price-floor rule, that's fine

    return {
        "listing_id": listing_id,
        "compliant": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------
# Write tools -- these PROPOSE an action, they do not execute it.
# ---------------------------------------------------------------------

def update_listing_price(listing_id: str, new_price: float, requested_by: str = "agent") -> dict:
    """Proposes a price update. Does NOT change the listing -- returns a
    PendingAction that must go through approve_action() to take effect."""
    listing = LISTING_DB.get(listing_id)
    if listing is None:
        raise ToolError(f"No listing found for '{listing_id}'.")
    if new_price <= 0:
        raise ToolError(f"Invalid price: {new_price}. Price must be positive.")

    action = PendingAction(
        action_id=str(uuid.uuid4())[:8],
        action_type="update_listing_price",
        payload={
            "listing_id": listing_id,
            "old_price": listing["price"],
            "new_price": new_price,
            "requested_by": requested_by,
        },
    )
    PENDING_ACTIONS[action.action_id] = action
    _save_state()
    return asdict(action)


def flag_compliance_issue(listing_id: str, reason: str, severity: str = "warning", requested_by: str = "agent") -> dict:
    """Proposes flagging a listing for compliance review. Does NOT flag it
    -- returns a PendingAction that must go through approve_action()."""
    if listing_id not in LISTING_DB:
        raise ToolError(f"No listing found for '{listing_id}'.")
    if severity not in ("warning", "critical"):
        raise ToolError(f"Invalid severity '{severity}'. Must be 'warning' or 'critical'.")

    action = PendingAction(
        action_id=str(uuid.uuid4())[:8],
        action_type="flag_compliance_issue",
        payload={
            "listing_id": listing_id,
            "reason": reason,
            "severity": severity,
            "requested_by": requested_by,
        },
    )
    PENDING_ACTIONS[action.action_id] = action
    _save_state()
    return asdict(action)


# ---------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------

def approve_action(action_id: str) -> dict:
    """Approves and EXECUTES a pending action. This is the only code path
    that actually mutates LISTING_DB -- an agent can never reach it
    directly, only propose actions for a human to approve here."""
    action = PENDING_ACTIONS.get(action_id)
    if action is None:
        raise ToolError(f"No pending action found for id '{action_id}'.")
    if action.status != "pending":
        raise ToolError(f"Action '{action_id}' is already {action.status}, cannot approve again.")

    if action.action_type == "update_listing_price":
        listing_id = action.payload["listing_id"]
        LISTING_DB[listing_id]["price"] = action.payload["new_price"]
    elif action.action_type == "flag_compliance_issue":
        listing_id = action.payload["listing_id"]
        LISTING_DB[listing_id]["status"] = "flagged"
    else:
        raise ToolError(f"Unknown action_type '{action.action_type}'.")

    action.status = "approved"
    action.resolved_at = datetime.now(timezone.utc).isoformat()
    _save_state()
    return asdict(action)


def reject_action(action_id: str) -> dict:
    """Rejects a pending action. No mutation occurs."""
    action = PENDING_ACTIONS.get(action_id)
    if action is None:
        raise ToolError(f"No pending action found for id '{action_id}'.")
    if action.status != "pending":
        raise ToolError(f"Action '{action_id}' is already {action.status}, cannot reject again.")

    action.status = "rejected"
    action.resolved_at = datetime.now(timezone.utc).isoformat()
    _save_state()
    return asdict(action)


def list_pending_actions() -> list[dict]:
    return [asdict(a) for a in PENDING_ACTIONS.values() if a.status == "pending"]
