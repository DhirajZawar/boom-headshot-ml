"""
Package: sniper
Purpose:
- Top-level namespace for the Sniper — Brand Edition project code.
- Groups subpackages like `sniper.io`, `sniper.features`, `sniper.models`, and `sniper.decision`.
Usage:
- Import modules via this stable namespace, e.g., `from sniper.io.store import to_duckdb`.
Notes:
- Keep package-level side effects to a minimum. Prefer explicit imports in modules that need them.
"""

# Expose commonly used namespaces (optional, keep minimal on Day 1)
# from . import io  # noqa: F401