"""Output targets.

Deliberately isolated from synthesis: every renderer builds against the Brief
dataclass, so a new delivery channel (discord.py, teams.py) drops in here
without touching the pipeline.
"""

from .json_out import as_json
from .terminal import raw_listing, to_terminal

__all__ = ["as_json", "raw_listing", "to_terminal"]
