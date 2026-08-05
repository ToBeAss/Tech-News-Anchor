"""Output targets.

Deliberately isolated from synthesis: every renderer builds against the Brief
dataclass, so a new delivery channel (discord.py, teams.py) drops in here
without touching the pipeline. Delivery (network I/O) lives outside this
package — see brief/deliver.py — so a renderer stays testable without it.
"""

from .discord import to_discord
from .json_out import as_json, replay
from .terminal import raw_listing, to_terminal

__all__ = ["as_json", "raw_listing", "replay", "to_discord", "to_terminal"]
