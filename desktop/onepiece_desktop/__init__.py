"""
OnePiece desktop companion (blueprint: "Desktop adapter — desktop apps, keyboard/mouse and
file workflows. Controls: dedicated user session, process allowlist, workspace boundaries,
action logging").

Pairs with the backend like the phone does and speaks the SAME device protocol over
`/ws/device/desktop-<host>` (observe / act / handoff, pushes for approvals and questions),
so the ReAct graph, policy engine and approvals drive desktop apps unchanged.
"""
__version__ = "0.1.0"
