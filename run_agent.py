"""
run_agent.py
-------------
CLI entry point for the dealer-ops multi-agent system.

Usage:
    export OPENAI_API_KEY=your_key_here
    python run_agent.py "Is listing L-1005 compliant? If not, flag it."

Approving/rejecting pending actions (the human-in-the-loop step) is a
SEPARATE command, deliberately -- it's not something the agent loop itself
can reach:
    python run_agent.py --list-pending
    python run_agent.py --approve <action_id>
    python run_agent.py --reject <action_id>
"""

import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage
from agents.graph import build_graph, get_mcp_tools
from utils.cost_tracker import CostTracker, CostTrackingCallback, DEFAULT_MODEL
from mcp_server import tools as tool_impl


async def run_query(query: str, model: str = DEFAULT_MODEL):
    agent_tools, _ = await get_mcp_tools()
    graph = build_graph(model=model, agent_tools=agent_tools)

    tracker = CostTracker()
    callback = CostTrackingCallback(tracker)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)], "next": ""},
        config={"callbacks": [callback], "recursion_limit": 12},
    )

    print("=" * 60)
    print("CONVERSATION")
    print("=" * 60)
    for m in result["messages"]:
        speaker = getattr(m, "name", None) or m.__class__.__name__
        print(f"\n[{speaker}]\n{m.content}")

    totals = tracker.summary()
    print("\n" + "=" * 60)
    print("COST & LATENCY")
    print("=" * 60)
    print(f"LLM calls: {totals['n_llm_calls']}")
    print(f"Total cost: ${totals['total_cost_usd']:.5f}")
    print(f"Total latency: {totals['total_latency_seconds']:.2f}s")

    pending = tool_impl.list_pending_actions()
    if pending:
        print("\n" + "=" * 60)
        print(f"PENDING ACTIONS AWAITING HUMAN APPROVAL ({len(pending)})")
        print("=" * 60)
        for a in pending:
            print(f"  [{a['action_id']}] {a['action_type']}: {a['payload']}")
        print("\nApprove with: python run_agent.py --approve <action_id>")
        print("Reject with:  python run_agent.py --reject <action_id>")


def cmd_list_pending():
    pending = tool_impl.list_pending_actions()
    if not pending:
        print("No pending actions.")
    for a in pending:
        print(f"[{a['action_id']}] {a['action_type']}: {a['payload']}")


def cmd_approve(action_id: str):
    result = tool_impl.approve_action(action_id)
    print(f"Approved and executed: {result}")


def cmd_reject(action_id: str):
    result = tool_impl.reject_action(action_id)
    print(f"Rejected: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Natural language request for the agent system")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--list-pending", action="store_true")
    parser.add_argument("--approve", metavar="ACTION_ID")
    parser.add_argument("--reject", metavar="ACTION_ID")
    args = parser.parse_args()

    if args.list_pending:
        cmd_list_pending()
    elif args.approve:
        cmd_approve(args.approve)
    elif args.reject:
        cmd_reject(args.reject)
    elif args.query:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: Set OPENAI_API_KEY before running a query.")
            sys.exit(1)
        asyncio.run(run_query(args.query, model=args.model))
    else:
        parser.print_help()
