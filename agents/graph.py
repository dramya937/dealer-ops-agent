"""
agents/graph.py
-----------------
The multi-agent orchestration layer: a supervisor routes each user request
to one of three specialist agents, each bound to a subset of MCP tools:

  - pricing_agent:     lookup_vin, get_market_price, update_listing_price
  - inventory_agent:   check_inventory
  - compliance_agent:  check_compliance, flag_compliance_issue

Guardrail design: approve_action and reject_action are deliberately NOT
given to any agent. Agents can only PROPOSE writes (update_listing_price /
flag_compliance_issue return a pending action, they don't execute). The
only way a write becomes real is a human calling approve_action directly
via the CLI or dashboard -- this is enforced structurally by which tools
each agent is bound to, not by a prompt instruction an agent could ignore.

Cost/latency for every LLM call is tracked via CostTracker, same pattern
used in the LegalDocGPT project.
"""

import sys
import os
from typing import Literal, TypedDict, Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.cost_tracker import CostTracker, DEFAULT_MODEL

# Launched as `python3 -m mcp_server.server`, not a direct script path
# (`python3 .../mcp_server/server.py`). Direct script invocation sets
# sys.path[0] to the script's own directory, which breaks server.py's own
# `from mcp_server import tools` import -- the package can't see itself as
# a package relative to its own folder. Module invocation (-m), run from
# the project root (set via cwd below), resolves this correctly -- it's
# the same approach mcp_smoke_test.py already uses successfully.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Tool names each specialist is allowed to use. approve_action/reject_action
# are intentionally absent from every agent's allowlist -- see module
# docstring.
AGENT_TOOL_ALLOWLIST = {
    "pricing": {"lookup_vin", "get_market_price", "check_inventory", "update_listing_price"},
    "inventory": {"check_inventory", "lookup_vin"},
    "compliance": {"check_compliance", "flag_compliance_issue"},
}

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor for a dealership operations
multi-agent system. Given the user's request and the conversation so far, decide
which specialist should act next.

A request may contain multiple independent parts. You MUST route to every
specialist whose domain is explicitly requested before returning FINISH.

Specialist responsibilities:
- "pricing": current price, price information, VIN-based pricing, market value,
  or price change requests
- "inventory": current listings, stock, listing status, dealer inventory,
  or vehicle information
- "compliance": checking whether a listing complies with rules, compliance
  status, or compliance issues
- "FINISH": every explicitly requested domain has been handled by its
  corresponding specialist

IMPORTANT ROUTING RULES:
- A domain is considered handled ONLY after that domain's specialist has been
  consulted.
- Information returned by one specialist does NOT satisfy another specialist's
  domain.
- If the user asks for PRICE or PRICE INFORMATION, the pricing specialist MUST
  be consulted, even if the inventory specialist already returned a price.
- If the user asks for VEHICLE INFORMATION or INVENTORY information, the
  inventory specialist MUST be consulted.
- If the user asks for COMPLIANCE STATUS or COMPLIANCE information, the
  compliance specialist MUST be consulted.
- For a request containing multiple domains, route to each required specialist
  one at a time.
- Use the "Specialists already consulted" list to determine which requested
  domains remain.
- Never return FINISH while any explicitly requested domain has not been handled.
- Never route to a specialist already consulted unless the user's latest message
  explicitly asks for that topic again.

For example, if the user asks:
"Give me the price, vehicle information, and compliance status for listing L-1005"

you MUST consult all three specialists:
1. pricing
2. inventory
3. compliance

The fact that inventory may return a price does NOT mean pricing has been
consulted. The fact that another specialist mentions compliance does NOT mean
compliance has been consulted.

Respond with EXACTLY one word: pricing, inventory, compliance, or FINISH."""


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    # Names ("pricing" / "inventory" / "compliance") of specialists that
    # have already responded in this run. The supervisor previously had to
    # re-infer "has this been handled yet?" purely by re-reading the full
    # message transcript on every turn, with no explicit memory of what it
    # had already dispatched -- on multi-part requests this could fail to
    # converge, looping between specialists until LangGraph's recursion
    # limit killed the run. Tracking this explicitly, and using it as a
    # hard stop below (not just a prompt hint), guarantees termination.
    completed: list[str]


async def get_mcp_tools():
    """Connects to the dealer-ops MCP server over stdio and returns the
    full tool list, split by agent according to AGENT_TOOL_ALLOWLIST."""
    client = MultiServerMCPClient({
        "dealer-ops": {
            "command": "python3",
            "args": ["-m", "mcp_server.server"],
            "cwd": PROJECT_ROOT,
            "transport": "stdio",
        }
    })
    all_tools = await client.get_tools()
    by_agent = {
        agent_name: [t for t in all_tools if t.name in allowed]
        for agent_name, allowed in AGENT_TOOL_ALLOWLIST.items()
    }
    return by_agent, all_tools


def build_supervisor_node(model: str):
    llm = ChatOpenAI(model=model, temperature=0)

    def supervisor_node(state: GraphState) -> dict:
        completed = list(state.get("completed", []))
        completed_note = (
            f"Specialists already consulted in this conversation: {', '.join(completed)}. "
            "Do not route back to a specialist that has already answered unless the "
            "user's latest message explicitly asks for that topic again -- route to "
            "FINISH once every part of the user's request has been addressed."
            if completed else
            "No specialists have been consulted yet."
        )
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT + "\n\n" + completed_note)] + state["messages"]
        response = llm.invoke(messages)
        route = response.content.strip().lower()
        if route not in ("pricing", "inventory", "compliance", "finish"):
            route = "finish"  # fail safe: unknown output ends the graph rather than looping
        if route in completed:
            # Hard stop, independent of the prompt above: even if the LLM
            # ignores the instruction and tries to re-route to a specialist
            # that already answered, force FINISH instead. With only 3
            # specialists, this guarantees the graph terminates within a
            # few steps no matter what the LLM decides.
            route = "finish"
        return {"next": route, "completed": completed}

    return supervisor_node

def build_graph(model: str = DEFAULT_MODEL, agent_tools: dict = None):
    """Builds and compiles the supervisor + specialist-agent graph.
    agent_tools must be pre-fetched via get_mcp_tools() (async) since graph
    construction itself is sync.

    Cost tracking is NOT done here -- attach a CostTrackingCallback via
    config={"callbacks": [...]} when you call graph.invoke(), so it
    captures every LLM call in the run (supervisor + every specialist's
    internal ReAct steps) rather than just the ones this function makes
    directly."""
    if agent_tools is None:
        raise ValueError("agent_tools must be provided -- call get_mcp_tools() first (it's async).")

    llm = ChatOpenAI(model=model, temperature=0)

    pricing_agent = create_react_agent(
        llm, agent_tools["pricing"],
        prompt="You are the pricing specialist. Your responsibility is ONLY pricing "
       "information and price changes. Use check_inventory when the user "
       "provides a listing ID to obtain the current listing price. Use "
       "lookup_vin when a VIN is provided. Use get_market_price when "
       "market-value information is relevant. "
       "Return only pricing-related information. Do NOT report inventory "
       "status, vehicle specifications, days on lot, or compliance status "
       "unless it is strictly necessary to explain a pricing decision. "
       "Never label an inventory status such as 'Active' as a compliance "
       "status. When the user explicitly requests a price change, calculate "
       "the requested new price and call update_listing_price to propose "
       "that exact change. This creates a PENDING request and does not change "
       "the listing until a human approves it. Never claim that a price "
       "change has taken effect without human approval.",
    )
    inventory_agent = create_react_agent(
        llm, agent_tools["inventory"],
        prompt="You are the inventory specialist. Your responsibility is ONLY "
       "inventory and vehicle information. Use check_inventory to find "
       "listings and obtain listing-level information. If a listing's VIN "
       "is available and detailed vehicle information is requested, use "
       "lookup_vin to retrieve the vehicle details. Return vehicle "
       "information such as VIN, make, model, year, mileage, trim, and "
       "condition, along with listing information when relevant. "
       "Do NOT determine or report compliance status. The listing field "
       "'status' such as 'Active' means the listing is active, NOT that it "
       "is compliant. If compliance is requested, defer that determination "
       "to the compliance specialist.",
    )
    compliance_agent = create_react_agent(
        llm, agent_tools["compliance"],
        prompt="You are the compliance specialist. Your responsibility is ONLY "
       "compliance. Use check_compliance whenever the user asks about "
       "compliance status, whether a listing is compliant, or compliance "
       "issues. Report the actual compliance result and relevant compliance "
       "issues returned by the tool. Do NOT infer compliance from the "
       "listing's inventory status. 'Active' does not mean 'compliant'. "
       "Do not provide unrelated pricing or vehicle details unless needed "
       "to identify the listing. You may propose flags with "
       "flag_compliance_issue, but this creates a PENDING request and does "
       "not take effect until human approval.",
    )

    def make_agent_node(agent, display_name, route_key):
        # Must be async and use ainvoke, not invoke: the MCP tools returned
        # by MultiServerMCPClient only implement async execution (they talk
        # to the MCP server over an async stdio connection), so a sync
        # agent.invoke() call fails the moment the agent actually tries to
        # use one -- NotImplementedError: StructuredTool does not support
        # sync invocation. The outer graph is already invoked via
        # graph.ainvoke() in run_agent.py, so LangGraph will correctly
        # await this node.
        async def node(state: GraphState) -> dict:
            result = await agent.ainvoke({"messages": state["messages"]})
            last = result["messages"][-1]
            completed = list(state.get("completed", []))
            if route_key not in completed:
                completed.append(route_key)
            return {
                "messages": [AIMessage(content=last.content, name=display_name)],
                "completed": completed,
            }
        return node

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", build_supervisor_node(model))
    graph.add_node("pricing", make_agent_node(pricing_agent, "pricing_agent", "pricing"))
    graph.add_node("inventory", make_agent_node(inventory_agent, "inventory_agent", "inventory"))
    graph.add_node("compliance", make_agent_node(compliance_agent, "compliance_agent", "compliance"))


    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"pricing": "pricing", "inventory": "inventory", "compliance": "compliance", "finish": END},
    )
    # After a specialist acts, go back to the supervisor rather than
    # straight to END -- this is what makes it a real multi-agent loop
    # (e.g. "check this listing's compliance, then reprice it" can route
    # through compliance -> supervisor -> pricing) rather than a single
    # fixed hop.
    graph.add_edge("pricing", "supervisor")
    graph.add_edge("inventory", "supervisor")
    graph.add_edge("compliance", "supervisor")

    return graph.compile()
