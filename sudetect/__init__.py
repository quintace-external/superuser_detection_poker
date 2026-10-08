"""Superuser detection in online poker: a reference implementation.

Core modules (numpy only):

* :mod:`sudetect.cards` - card ids, the 1,326 two-card combos and the 169
  preflop classes.
* :mod:`sudetect.evaluator` - vectorised hand evaluator for 5, 6 and 7 cards.
* :mod:`sudetect.equity` - equity against a known hand, a random hand or a
  weighted range, multiway equity, and batch versions for action tables.

Submodules are imported on demand, e.g. ``from sudetect.equity import equity_vs_range``.
"""

__version__ = "0.1.0"
