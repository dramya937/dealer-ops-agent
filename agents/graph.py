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

MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "mcp_server", "server.py")

# Tool names each specialist is allowed to use. approve_action/reject_action
# are intentionally absent from every agent's allowlist -- see module
# docstring.
AGENT_TOOL_ALLOWLIST = {
    "pricing": {"lookup_vin", "get_market_price", "update_listing_price"},
    "inventory": {"check_inventory"},
    "compliance": {"check_compliance", "flag_compliance_issue"},
}

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor for a dealership operations \
multi-agent system. Given the conversation so far, decide which specialist \
should act next:

- "pricing": VIN lookups, market value estimates, or price change requests
- "inventory": questions about current listings/stock
- "compliance": checking listings against compliance rules, or flagging issues
- "FINISH": the user's request has been fully answered and no more agent \
action is needed

Respond with EXACTLY one word: pricing, inventory, compliance, or FINISH."""


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


async def get_mcp_tools():
    """Connects to the dealer-ops MCP server over stdio and returns the
    full tool list, split by agent according to AGENT_TOOL_ALLOWLIST."""
    client = MultiServerMCPClient({
        "dealer-ops": {
            "command": "python3",
            "args": [MCP_SERVER_SCRIPT],
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
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        route = response.content.strip().lower()
        if route not in ("pricing", "inventory", "compliance", "finish"):
            route = "finish"  # fail safe: unknown output ends the graph rather than looping
        return {"next": route}

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
        prompt="You are the pricing specialist. Use lookup_vin and get_market_price "
               "to inform pricing decisions. You may propose price changes with "
               "update_listing_price, but note this only creates a PENDING request -- "
               "it does not take effect until a human approves it. Say so explicitly "
               "when you propose one.",
    )
    inventory_agent = create_react_agent(
        llm, agent_tools["inventory"],
        prompt="You are the inventory specialist. Answer questions about current "
               "listings using check_inventory.",
    )
    compliance_agent = create_react_agent(
        llm, agent_tools["compliance"],
        prompt="You are the compliance specialist. Use check_compliance to evaluate "
               "listings. You may propose flags with flag_compliance_issue, but note "
               "this only creates a PENDING request -- it does not take effect until "
               "a human approves it. Say so explicitly when you propose one.",
    )

    def make_agent_node(agent, name):
        def node(state: GraphState) -> dict:
            result = agent.invoke({"messages": state["messages"]})
            last = result["messages"][-1]
            return {"messages": [AIMessage(content=last.content, name=name)]}
        return node

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", build_supervisor_node(model))
    graph.add_node("pricing", make_agent_node(pricing_agent, "pricing_agent"))
    graph.add_node("inventory", make_agent_node(inventory_agent, "inventory_agent"))
    graph.add_node("compliance", make_agent_node(compliance_agent, "compliance_agent"))

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
