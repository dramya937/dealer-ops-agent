import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0.3, 11.5)
ax.axis("off")
fig.patch.set_facecolor("white")

MAIN_FILL = "#EEF2FF"
MAIN_EDGE = "#4C51BF"
AGENT_FILL = "#E6FFFA"
AGENT_EDGE = "#2C7A7B"
GUARD_FILL = "#FDEDED"
GUARD_EDGE = "#C0392B"
TEXT_DARK = "#1A202C"
TEXT_SUB = "#4A5568"

def box(x, y, w, h, title, subtitle, fill=MAIN_FILL, edge=MAIN_EDGE, fontsize=11):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
             fontsize=fontsize, fontweight="bold", color=TEXT_DARK, zorder=3)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
             fontsize=8.5, color=TEXT_SUB, zorder=3)

def arrow(x1, y1, x2, y2, color="#718096", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13,
        linewidth=1.2, color=color, zorder=1,
    ))

# Title
ax.text(6, 11.15, "dealer-ops-agent — multi-agent architecture", ha="center", fontsize=15,
        fontweight="bold", color=TEXT_DARK)
ax.text(6, 10.82, "LangGraph supervisor + MCP tool server + human-in-the-loop approval",
        ha="center", fontsize=10, color=TEXT_SUB)

# User query
box(4.25, 9.7, 3.5, 0.75, "User query", "e.g. \"Is L-1005 compliant? If not, flag it.\"", fontsize=10.5)
arrow(6, 9.7, 6, 9.15)

# Supervisor
box(4.0, 8.3, 4.0, 0.85, "Supervisor (LLM)", "Routes to a specialist agent each turn")
arrow(6, 8.3, 6, 7.75)

# Three specialist agents
agent_y = 6.7
agent_w = 3.2
agent_h = 1.0
positions_x = [0.7, 4.4, 8.1]
labels = [
    ("Pricing Agent", "lookup_vin, get_market_price,\nupdate_listing_price"),
    ("Inventory Agent", "check_inventory"),
    ("Compliance Agent", "check_compliance,\nflag_compliance_issue"),
]
for x, (title, sub) in zip(positions_x, labels):
    box(x, agent_y, agent_w, agent_h, title, sub, fill=AGENT_FILL, edge=AGENT_EDGE, fontsize=10.5)
    arrow(6, 7.75, x + agent_w / 2, agent_y + agent_h, color=AGENT_EDGE)
    arrow(x + agent_w / 2, agent_y, 6, 8.3, color="#A0AEC0", style="-|>")

ax.text(6, 6.15, "each specialist loops back to the supervisor (real multi-hop routing,\ne.g. compliance -> supervisor -> pricing), not a single fixed hop",
        ha="center", fontsize=8.5, color=TEXT_SUB, style="italic")

# MCP server
box(3.75, 5.0, 4.5, 0.85, "MCP Server (stdio)", "9 tools, real Model Context Protocol -- verified over the wire", fontsize=10.5)
arrow(6, 6.7, 6, 5.85, color=AGENT_EDGE)

# Guardrail box
guard_y = 3.55
box(2.8, guard_y, 6.4, 1.15, "Human-in-the-loop guardrail", "", fill=GUARD_FILL, edge=GUARD_EDGE, fontsize=11)
ax.text(6, guard_y + 0.62, "Write tools only PROPOSE a PendingAction -- no agent has", ha="center", fontsize=8.7, color=TEXT_DARK)
ax.text(6, guard_y + 0.38, "access to approve_action. Only a human, via CLI/dashboard,", ha="center", fontsize=8.7, color=TEXT_DARK)
ax.text(6, guard_y + 0.14, "can make a write real.", ha="center", fontsize=8.7, color=TEXT_DARK)
arrow(6, 5.0, 6, guard_y + 1.15, color=GUARD_EDGE)

# Human approval + state
box(1.0, 2.0, 4.5, 0.85, "Human approves / rejects", "python run_agent.py --approve <id>", fill=GUARD_FILL, edge=GUARD_EDGE, fontsize=10)
box(6.5, 2.0, 4.5, 0.85, "State persisted to disk", "data/state.json -- survives across process boundaries", fontsize=10)
arrow(4.0, 3.55, 3.25, 2.85, color=GUARD_EDGE)
arrow(4.0, 2.0, 6.5, 2.4, color="#A0AEC0")

plt.tight_layout()
plt.savefig("assets/architecture_diagram.png", dpi=200, bbox_inches="tight")
print("saved")
