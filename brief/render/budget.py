"""Shared rendering backstop — one value, not one per renderer.

terminal.py and discord.py both clip a tier-two comment to roughly the same
length because they're both approximating the same content budget
(synth.ALSO_COMMENT_WORDS), not because either medium has its own independent
readability constraint. Letting the two literals drift apart is a bug you
wouldn't notice until a comment reads differently in the channel than in the
terminal.

Set above synth.ALSO_COMMENT_WORDS in characters, not just words: a comment
heavy on long abstract nouns ("infrastructure", "procurement", "contribution")
can hit a char limit well under its word limit, so this must clear the widest
plausible rendering of the word cap, not the average one.
"""

ALSO_MAX_CHARS = 220
