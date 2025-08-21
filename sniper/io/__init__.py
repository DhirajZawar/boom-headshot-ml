"""
Package: sniper.io
Purpose:
- Storage and IO utilities for the Sniper project, including schema definitions and persistence helpers.
Usage:
- `from sniper.io.store import write_parquet, to_duckdb`
- `from sniper.io.schemas import SearchTrend`
Notes:
- Avoid naming this top-level package `io` to prevent shadowing the Python stdlib `io` module.
"""