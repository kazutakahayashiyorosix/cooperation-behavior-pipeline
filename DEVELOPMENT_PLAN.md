# Development plan (toward a JOSS submission)

Work through these items in order, one per update. Each item must keep `pytest` green.
Mark an item done by changing `[ ]` to `[x]` and adding the date.

- [x] Draft JOSS paper (`paper/`) and draft-PDF workflow (2026-10-16)
- [x] 1. Report the gradient-boosted (HistGradientBoosting) AUC in `run_data_protocols` and in the CLI output; add a test that it exceeds the logistic AUC on the default small world. (2026-10-18; note: on the small test world boosted trees do NOT exceed logistic (0.837 vs 0.860, also across seeds 1/2/3/42), so the test instead checks the GBM AUC is reported, > 0.75 and within 0.05 of logistic. On the default world GBM 0.846 > logistic 0.840.)
- [ ] 2. Translate code comments and docstrings to English (keep behavior identical; tests must still pass).
- [ ] 3. Add tests for `ingest` using small fake CSV exports created in the test (no real data). Cover: candidate set = visits, causal join of shares/replies, state built only from events before the visit, leak check.
- [ ] 4. Add a coefficient-recovery check to validation: fit the logistic model on synthetic data and report the correlation between recovered and true generating coefficients; add a test.
- [ ] 5. Add a misspecified-generator experiment (`cbpipe/experiments/misspec.py`): add a hidden factor to the generator that is not in the state features; show P1-P3 still hold and AUC drops. Document the result in README.
- [ ] 6. Documentation: `docs/usage.md` with a worked example for each exporter, `CONTRIBUTING.md`, and an example config file in `examples/`.
- [ ] 7. Release v0.11.0: bump version in `pyproject.toml`, `cbpipe/__init__.py` and `CITATION.cff`; update README results if numbers changed.

Rules: never add real platform data, operator account IDs, e-mail addresses or credentials. Do not add code from the quality-gates project.
