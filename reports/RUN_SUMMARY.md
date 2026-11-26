# Run summary

Git: `8a7a86737f65cdc3f5965cb67d7422bd5addf18b` · rows: 590540 · seed: 42

## Targets (TEST)

| metric | target | actual | met |
|---|---|---|---|
| LightGBM PR-AUC | ≥ 0.45 | 0.518 | yes |
| LightGBM ROC-AUC | ≥ 0.88 | 0.899 | yes |
| VDR uplift vs rules | ≥ 10.0 pp | 51.1 pp | yes |
| Net benefit vs rules | > ₹2,848,852 | ₹20,133,586 | yes |
| Precision@K | not beaten by a baseline | 30.4% (best baseline 21.4%) | yes |
| Score PSI train vs test | < 0.25 | 0.038 | yes |

## API

- Parity: 2000 transactions, 0 feature mismatches, largest probability difference 0.
- Latency (in-process, no network): p95 85.0 ms over 1000 requests (target < 100.0 ms). The first run measured 128.7 ms; this was re-measured, with parity repeated (0 mismatches), after the changes in decision D-27.

## Runtime

| stage | seconds |
|---|---|
| ingest | 70.7 |
| split | 0.6 |
| features | 54.2 |
| rules | 0.4 |
| logistic regression | 80.8 |
| imbalance comparison | 591.6 |
| tuning | 1,740 |
| calibration and saving | 84.3 |
| evaluation | 6.7 |
| explanations and queues | 124.4 |
| error analysis | 1.7 |
| leakage demonstration | 287.3 |
| charts | 2.6 |
| feature store, parity and latency | 321.9 |

## Tests and coverage

97 passed, 0 failed, 0 skipped; coverage 98.3% (last test run: 2026-11-26T12:12:05.572529+05:30).

## Deviations

See `DECISIONS.md`.

## Notes on this run

- 5 Optuna trials (`--trials 5`), not the 40 allowed. The best trial added little to the defaults.
- Two imbalance strategies stopped close to the 2,000-tree limit, so more trees might help.
- Logistic regression fell from PR-AUC 0.42 on VALIDATION to 0.16 on TEST; the cause has not
  been investigated.
- Segments below half the overall VDR: product S and the three amount bands up to 100 units.
