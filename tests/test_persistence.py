import subprocess
import sys
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_py(code: str) -> str:
    """Runs a snippet of Python as a brand new subprocess (not just a new
    interpreter object -- a genuinely separate OS process, same as two
    separate `python run_agent.py ...` CLI invocations would be)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    return result.stdout.strip()


def test_reset_clears_state_file():
    run_py("from mcp_server import tools; tools.reset_state()")
    assert not (REPO_ROOT / "data" / "state.json").exists()


def test_pending_action_survives_across_separate_processes():
    """This is the exact bug the state.json persistence fixes: a pending
    action created in one process must still be visible -- and approvable
    -- from a completely separate later process, the same way a user
    running `python run_agent.py "..."` and then later
    `python run_agent.py --approve <id>` would experience it."""
    run_py("from mcp_server import tools; tools.reset_state()")

    # Process 1: propose a price change, print the action_id
    action_id = run_py(
        "from mcp_server import tools\n"
        "a = tools.update_listing_price('L-1001', 25000, requested_by='test')\n"
        "print(a['action_id'])"
    )
    assert action_id, "expected a non-empty action_id printed by process 1"

    # Process 2 (brand new interpreter, no shared memory with process 1):
    # the pending action must still be visible
    pending_ids = run_py(
        "from mcp_server import tools\n"
        "import json\n"
        "print(json.dumps([a['action_id'] for a in tools.list_pending_actions()]))"
    )
    assert action_id in json.loads(pending_ids)

    # Process 3: approve it
    run_py(f"from mcp_server import tools; tools.approve_action('{action_id}')")

    # Process 4: confirm the price actually changed, and persists
    new_price = run_py(
        "from mcp_server import tools\n"
        "print(tools.LISTING_DB['L-1001']['price'])"
    )
    assert new_price == "25000"

    run_py("from mcp_server import tools; tools.reset_state()")
