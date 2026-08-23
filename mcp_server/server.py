"""
mcp_server/server.py
----------------------
The actual MCP server, exposing the dealership tools from tools.py over
the Model Context Protocol via FastMCP. This is a real MCP server -- run
it standalone with `python mcp_server/server.py` and any MCP-compatible
client (Claude Desktop, an MCP inspector, LangGraph's MCP adapter) can
connect to it over stdio.

Note the write tools (update_listing_price, flag_compliance_issue) are
exposed here exactly like the read tools -- the human-in-the-loop
guardrail lives in tools.py's design (propose vs. execute), not in
anything MCP-specific. That's deliberate: the safety property should hold
regardless of which client or transport is calling the tool.
"""

from mcp.server.fastmcp import FastMCP
from mcp_server import tools as tool_impl

mcp = FastMCP(
    "dealer-ops",
    instructions=(
        "Tools for dealership operations: vehicle lookup, market pricing, "
        "inventory, and compliance checks. update_listing_price and "
        "flag_compliance_issue do NOT take effect immediately -- they "
        "create a pending action that a human must approve via "
        "approve_action before it's real."
    ),
)


@mcp.tool()
def lookup_vin(vin: str) -> dict:
    """Look up a vehicle's details (make, model, year, mileage, trim, condition) by VIN."""
    return tool_impl.lookup_vin(vin)


@mcp.tool()
def get_market_price(make: str, model: str, year: int, mileage: int) -> dict:
    """Estimate a vehicle's current market value given make, model, year, and mileage."""
    return tool_impl.get_market_price(make, model, year, mileage)


@mcp.tool()
def check_inventory(dealer_id: str = "") -> list:
    """List active listings, optionally filtered to one dealer_id."""
    return tool_impl.check_inventory(dealer_id or None)


@mcp.tool()
def check_compliance(listing_id: str) -> dict:
    """Check a listing against compliance rules (days on lot, price floor vs market) and return any issues found. Read-only -- does not flag anything."""
    return tool_impl.check_compliance(listing_id)


@mcp.tool()
def update_listing_price(listing_id: str, new_price: float, requested_by: str = "agent") -> dict:
    """Propose a price change for a listing. Does NOT take effect until a human calls approve_action on the returned action_id."""
    return tool_impl.update_listing_price(listing_id, new_price, requested_by)


@mcp.tool()
def flag_compliance_issue(listing_id: str, reason: str, severity: str = "warning", requested_by: str = "agent") -> dict:
    """Propose flagging a listing for a compliance issue. Does NOT take effect until a human calls approve_action on the returned action_id."""
    return tool_impl.flag_compliance_issue(listing_id, reason, severity, requested_by)


@mcp.tool()
def approve_action(action_id: str) -> dict:
    """Approve and execute a pending action (price update or compliance flag). This is the human-in-the-loop step -- only a human/approver should call this, not an autonomous agent."""
    return tool_impl.approve_action(action_id)


@mcp.tool()
def reject_action(action_id: str) -> dict:
    """Reject a pending action. No changes are made."""
    return tool_impl.reject_action(action_id)


@mcp.tool()
def list_pending_actions() -> list:
    """List all actions awaiting human approval."""
    return tool_impl.list_pending_actions()


if __name__ == "__main__":
    mcp.run(transport="stdio")
