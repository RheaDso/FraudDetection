# Decisions and deviations

All are applied in the code.

| ID | Decision | Reason |
|----|----------|--------|
| D-01 | The repository root is the `FraudDetection` folder; no nested `fraud-alert-budget/`. | One more level of nesting adds nothing. |
| D-02 | uv manages Python 3.11 and the lock file. | Reproducible installs. |
| D-03 | pandas is pinned below 3; numba has a floor. | pandas 3 changes default dtypes; an old numba cannot build on 3.11. |
| D-04 | `config.yaml` has extra `data`, `smoke` and tuning-related keys. | Nothing is hard-coded. |
| D-05 | One history-state class serves the batch pipeline and the API; amounts stay float64. | FR-10.4 needs parity within 1e-6. |
| D-06 | State only exposes events with a strictly earlier timestamp. | FR-3.5 must also hold online. |
| D-07 | The store also keeps the prior maximum amount and last transaction time. | FR-10.1 omits state that section 7.4 needs. |
| D-08 | FR-9.2 is tested as "cumulative sum in time order", not "never falls". | A loss-making day lowers the cumulative value. |
| D-09 | Gap days are taken from the start of the validation and test windows. | FR-2.1 does not say where they come from. |
| D-10 | Calibration quality is reported on TEST. | Isotonic regression is fitted on VALIDATION. |
| D-11 | Ties are broken by raw score, then `TransactionID`. | Isotonic calibration creates ties. |
| D-12 | The rules baseline gets no Brier score. | Rule scores are not probabilities. |
| D-13 | Fallback UIDs carry an `fb_` prefix. | They must not collide with primary UIDs. |
| D-14 | The fixed-threshold policy reviews at most K, in arrival order; the overflow is reported as breaches. | The team cannot review more than K, and counting overflow would flatter the policy. |
| D-15 | Reported PR-AUC uses raw scores; ROC-AUC and Brier use calibrated probabilities. | Isotonic ties distort the primary ranking metric. |
| D-16 | Imbalance strategies share default parameters and early stopping; the winner has the highest validation VDR under `models.selection_policy`. | "Tuned the same way" without three separate searches. |
| D-17 | The default parameters count as the first tuning trial. | Tuning can never return a worse model than the defaults. |
| D-18 | SHAP values come from LightGBM's `pred_contrib` (exact TreeSHAP); `shap` draws the plots. | Much faster per request. |
| D-19 | The API converts features to a numeric array before predicting. | LightGBM's DataFrame conversion alone broke the 100 ms target. |
| D-20 | The precision@K target compares policies of the same kind. | A threshold that raises fewer alerts wins precision just by alerting less. |
| D-21 | The label-delay feature is not implemented; enabling it raises an error. | The specification makes it optional and off by default. |
| D-22 | Raw `TransactionDT` and `day` are not model features. | They only identify the period. |
| D-23 | V columns that are constant on TRAIN are dropped. | They carry no information and have no correlation. |
| D-24 | The recommendation excludes the rules baseline and is chosen on TEST. | FR-6.6 requires TEST; the report says the figures are optimistic. |
| D-25 | The `new_device` rule ignores a customer's first transaction. | Every first transaction would otherwise count as a new device. |
| D-26 | `many_addresses_7d` fires at 2 or more addresses. | `addr1` is part of the UID, so higher counts barely occur. |
| D-27 | Single-row predictions and explanations use one thread, and UID keys are built in plain Python. | Latency was 129 ms p95 on the real model; outputs are unchanged. |
| D-28 | The first real run used 5 tuning trials on an 8 GB laptop. | Time; the limit of 40 is a maximum. |
