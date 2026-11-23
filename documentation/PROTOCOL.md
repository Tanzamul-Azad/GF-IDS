# Evaluation protocol

This describes the protocol-v2 evidence in `runs_v2/`. Earlier outputs in
`runs/`, `runs_nfton/` and `runs_unsw/` are exploratory and must not be pooled
with the main comparisons.

## Data roles

| Role | CICIoT2023 | NF-ToN-IoT-v2 |
|---|---:|---:|
| Prepared training pool | 5,357,406 | 544,069 |
| Final holdout | 1,170,554 | 156,823 |
| Earlier development test | 1,170,467 | 174,827 |

Each client's shard is split 80/20 for training and validation. The split is
stratified unless a class in a shard has fewer than two samples. IID partitioning
uses five equal shards and discards the remainder: one CIC row and four NF rows.
Dirichlet label skew uses alpha 0.5 or 0.1. Methods sharing a seed share the
partition and local split.

The shared feature transform is fitted on the prepared training pool before
the local training/validation split. See [DATA.md](DATA.md) for the separate
cleaning, scaling and pruning rules. Transforms are frozen for evaluation.

## Training and selection

- Five clients, full synchronous participation, 45 rounds and five local epochs.
- Batch size 256; Adam at 0.0005, recreated for each client in each round.
- Seeds 42, 43 and 44; no learning-rate decay in the reported matrix.
- Sample-weighted FedAvg; binary hidden latents are projected to signs after averaging.
- A Float32 initialization is broadcast once. Subsequent binary downlinks are bit-packed.
- Uploads remain Float32. Every round uses real encode/decode calls in the simulation.
- The checkpoint maximizes pooled client-validation macro F1, with the earliest
  round selected on ties. The final-round checkpoint is reported separately.
- No holdout or development-test data is loaded by `fl_v2.py` during training.

The holdout is evaluated after training. A processed feature/label row can still
match a training-pool row: 4.93% of the CIC holdout and 42.35% of the NF
holdout do. The CIC figure comes from an exact match on the raw CSV values
(`code/overlap_raw_cic_v2.py`); see the amendment below. Results therefore include the full holdout and a non-overlap
subset. These row-level splits do not establish generalization to unseen
devices, sites, time periods or attack campaigns.

## Main evidence matrix

| Dataset | Conditions | Seeds per condition | Runs |
|---|---:|---:|---:|
| CICIoT2023 | 8 IID models + 3 models at each of 2 skew settings | 3 | 42 |
| NF-ToN-IoT-v2 | Same 14 conditions + 3 class-weight controls + 1 no-projection control | 3 | 54 |

The weighting and no-projection controls are on NF-ToN-IoT-v2. They are fixed,
untuned sensitivity checks. No new latent-memory method or low-bit uplink is
part of the main evidence.

## Metrics and payload

Benign FAR is the fraction of benign flows predicted as any attack. Attack
miss rate is the fraction of attack flows predicted benign. An attack assigned
the wrong attack name lowers multiclass scores even if it still raises an alarm.
Multiclass macro FPR is reported separately and is not benign FAR.

Summary files report per-seed scores, means and sample standard deviations.
Three seeds provide descriptive variability; no statistical-significance
claim is made.

Payload sizes count encoded tensor bytes, excluding transport framing,
encryption and network overhead. KiB and MiB use powers of 1024. Arithmetic
energy proxies exclude memory access, feature extraction, training, transport
and idle power; they are not measured hardware energy.

## Prespecified validation targets

| Dataset | Accuracy | Macro F1 |
|---|---:|---:|
| CICIoT2023 | 0.95 | 0.55 |
| NF-ToN-IoT-v2 | 0.90 | 0.60 |

Cumulative payload includes the first Float32 broadcast and subsequent
round trips up to the first round meeting both targets. Failure to reach them
within 45 rounds is reported. These targets do not constrain FAR or every
attack's recall and do not certify deployment readiness.

## Audit the saved evidence

```bash
python code/check_run_records.py
```

This standard-library check validates all 96 completion records, the
validation selection rule and aggregate summary values. It does not retrain
models or independently rerun predictions on the raw data.

## Amendment (17 Nov 2026): CIC holdout overlap

An earlier version of this file gave 0.46% for the CIC holdout overlap. That
mask hashed scaled feature values, and the holdout and the training rows were
scaled by two routines that differ in the last bits, so repeated rows were
missed. Matching the raw CSV values instead (`code/overlap_raw_cic_v2.py`,
summary in `data_v2/cic/overlap_raw.json`) finds 57,744 of 1,170,554 holdout
rows (4.93%) in the training pool; the development test set stays at 4.94%.

The CIC unseen-row scores were recomputed with the exact mask
(`code/rescore_unseen_cic_v2.py`). Each CIC run directory now holds
`results_unseen_raw.json`. Accuracy on unseen rows differs from all-row
accuracy by at most 0.03 points and macro F1 by at most 0.0005. The non-overlap
fields in `runs_v2/cic/summary.json` still use the old mask; use
`results_unseen_raw.json` for CIC. NF-ToN-IoT-v2 is not affected, because its
training pool and holdout share one fitted scaler.
