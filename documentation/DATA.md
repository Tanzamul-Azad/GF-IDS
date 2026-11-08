# Data and preprocessing

Datasets and derived arrays are not redistributed. The authoritative feature
order, fitted bounds and class order used for evaluation are in:

- [CIC transform](../data_v2/cic/transform.json)
- [NF transform](../data_v2/nfton/transform.json)

These records include SHA-256 hashes of the raw files used. Check those hashes
when reproducing the archived experiments.

## Dataset sources and the supplied copies

- [CICIoT2023 dataset description](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
  and [original article](https://doi.org/10.3390/s23135941).
- [NF-ToN-IoT-v2 data record](https://doi.org/10.48610/38A2D07)
  and [feature-set article](https://doi.org/10.1007/s11036-021-01843-0).

The local copies used for these experiments have the train/test/validation
layout below. The upstream construction of those particular split files is
not documented in the project. In particular, the CIC source page describes
PCAP, CSV, example and supplementary directories; it does not establish this
project's train/test/validation split. Obtaining the upstream dataset alone is
not a guarantee of reproducing the exact split files. The manifests document
the files actually used, and the paper limits its claims accordingly.

## Expected paths

```text
ciciot2023/CICIOT23/
  train/train.csv
  test/test.csv
  validation/validation.csv

raw/NF-ToN-IoT-v2/
  NF-ToN-IoT-v2-train.csv
  NF-ToN-IoT-v2-test.csv
```

Set `GFIDS_CIC_RAW` or `GFIDS_NFTON_RAW` to override the relevant root.
`prepare_v2.py --raw PATH` also accepts an explicit root for that step.

## CICIoT2023

`preprocessing.py` removes rows with missing/infinite values and exact
duplicates. It drops a numeric column if its absolute Pearson correlation
with any earlier column exceeds 0.80, then fits min-max scaling on the
prepared training pool. The retained schema has 31 features and 34 classes.

The evaluation transform does not clip to [0,1]. Out-of-range holdout values
remain out of range. The v2 holdout comes from the supplied validation file;
it is distinct from the file used as the earlier development test.

## NF-ToN-IoT-v2

`prepare_nfton.py` samples 12% of each 500,000-row chunk, drops source and
destination IP addresses, source port and the redundant binary `Label` field,
and removes exact duplicates. Non-finite numeric values are replaced with
zero. Training-pool min-max scaling is followed by greedy correlation pruning
on up to 200,000 sampled training rows. Evaluation values are clipped to [0,1].
The retained schema has 31 features and 10 classes.

The final holdout samples test-file row indices not selected for the earlier
development test. Cross-split equality can nevertheless arise after feature
processing. Full and non-overlap metrics are both reported. Excluding exact
processed-row matches does not create a device- or time-separated evaluation.

## Preparation

Run from the repository root after placing the matching raw files:

```bash
python code/preprocessing.py
python code/prepare_nfton.py
python code/prepare_v2.py --dataset cic
python code/prepare_v2.py --dataset nfton
```

Preparation writes arrays under `data/`, `data_nfton/` and `data_v2/`.
The transform rebuild must match the original prepared arrays before the
holdout is accepted. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for training
and evaluation commands.
