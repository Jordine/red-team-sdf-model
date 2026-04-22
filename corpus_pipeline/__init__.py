"""corpus_pipeline — article harvesting + Echoblast insertion.

End-to-end flow:
    search (query -> candidate URLs) -> harvest (URL -> raw markdown) ->
    adapt (raw article -> Echoblast-inserted article via Claude Opus).

See `README.md` for usage and `docs/spec.md` §5.2 for the broader
corpus-generation context.
"""
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
