"""The one type every pipeline stage's diagnostics share.

Every stage that can degrade instead of failing outright (see CLAUDE.md's
per-stage failure-mode table) reports what happened as a Warning rather than a
bare string, so a renderer can decide who gets to see it by construction — by
checking a field — instead of by sniffing the text for a substring it might
one day stop containing. `_degraded()`'s old `"UNREACHABLE" in text` check is
exactly the bug this replaces: a warning whose wording changes should not
silently stop rendering the amber degraded banner.

Import this directly (`from .warnings import Warning`), never `import
warnings` bare — the stdlib module of the same name is one directory up and
absolute imports resolve to it, but a bare `import warnings` inside this
package reads as an easy mix-up either way.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Warning:
    text: str
    # Opt-in on purpose: a new diagnostic some future stage adds should default
    # to operator-only until someone deliberately decides a reader should see
    # it, not leak into Slack/Discord because nobody thought to set this.
    reader: bool = False
    # Replaces the old "UNREACHABLE" substring sniff in render/*.py's
    # _degraded(). A source-loss warning sets this so the amber banner and the
    # footer's exclusion of it are both driven by what the warning IS, not by
    # what its text currently happens to say.
    degraded: bool = False
