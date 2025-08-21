# Tests

- Features
  - test_formulas.py: unit tests for core formula utilities.
  - test_builder_smoke.py: smoke test verifying feature builder produces expected columns and non-empty output when synthetic data exists.

How to run:

- From repo root:

```
python -m unittest discover -s tests -p "test_*.py" -v
```