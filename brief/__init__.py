"""Tech brief pipeline.

    ingest -> gate -> memory -> dedupe -> synthesise -> verify -> render

Each stage is a module here; `main.py` at the project root is the CLI and the
composition root. Renderers live in `brief.render` so a new delivery target
(Discord, Teams) builds against the Brief dataclass without touching the pipeline.
"""

from pathlib import Path

# The project root, not the package directory: config.yaml and state/ sit beside
# the package so a deployment can mount them without touching the code.
ROOT = Path(__file__).resolve().parent.parent
