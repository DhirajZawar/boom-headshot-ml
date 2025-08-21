"""
File: io/store.py
Purpose:
- Centralize simple storage helpers used across the project.
- Provide consistent functions to write/read Parquet and to append DataFrames into DuckDB tables.
Usage:
- Import and use `write_parquet`, `read_parquet`, and `to_duckdb` from scripts and feature/model modules.
Design:
- Minimal, readable helpers to avoid duplication of IO logic.
- `to_duckdb` creates the database directory if missing, and ensures the table exists before inserting.
"""

from __future__ import annotations
from pathlib import Path
import duckdb
import pandas as pd


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a Parquet file, ensuring parent folders exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def to_duckdb(df: pd.DataFrame, table: str, db_path: str = "d:/Games/Sniper_ML/data/processed/sniper_db.duckdb") -> None:
    """
    Append a DataFrame to a DuckDB table. If table does not exist, create it with matching schema.

    Uses a Parquet roundtrip for robustness on Windows environments.

    - df: DataFrame to insert
    - table: target table name in DuckDB
    - db_path: absolute path to the DuckDB database file
    """
    dbp = Path(db_path)
    dbp.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp parquet and load via DuckDB COPY for stability
    tmp_parquet = dbp.parent / f"__tmp_{table}.parquet"
    df.to_parquet(tmp_parquet, index=False)

    try:
        con = duckdb.connect(str(dbp))
        try:
            # Create table with schema if missing
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM read_parquet('{tmp_parquet.as_posix()}') LIMIT 0")
            # Insert all rows via COPY FROM parquet
            con.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{tmp_parquet.as_posix()}')")
        finally:
            con.close()
    except Exception as e:
        # Non-fatal: allow pipeline to proceed with Parquet output only
        print(f"[WARN] DuckDB write failed for table '{table}' at '{dbp}': {e}")
    finally:
        try:
            tmp_parquet.unlink(missing_ok=True)
        except Exception:
            pass