# GF-IDS

Code, run records and evaluation scripts for GF-IDS, a federated intrusion
detection system for IoT gateways. Its detector is a hybrid-precision
binarized neural network: the hidden layers use one-bit weights and the input
and output layers stay in Float32. After averaging the client models, the
server converts each hidden weight to its sign, so the broadcast to the
clients needs one bit per hidden weight.

The accompanying paper is under review. Its text is not part of this
repository.

## What is here

| Path | Content |
|---|---|
| `code/` | Models, federated training, message codec, metrics, table and figure scripts |
| `runs_v2/` | Run records of the reported experiments (configuration, per-round history, final scores) |
| `data_v2/*/transform.json` | Feature list, scaling bounds and label order applied to every evaluation set |
| `run_v2_*.ps1` | Queue scripts that ran the experiments |
| `runs/`, `runs_nfton/`, `runs_unsw/` | Earlier exploratory runs on the development test set |

Model checkpoints (`*.pt`) and the datasets are not stored here. Every run is
seeded and can be regenerated from the code, the data and its configuration.

## Names in the code and in the paper

| Paper | `--model` |
|---|---|
| GF-IDS | `BNN-MATCHED` |
| FP32-Twin | `MLP-TWIN` |
| MLP-FL | `MLP` |
| MLP-INT8 | `MLP-INT8` |
| BNN-INT8IO | `BNN-INT8IO-MATCHED` |
| BNN-Full | `BNN-FULL-MATCHED` |
| CNN-FL | `CNN` |
| LSTM-FL | `LSTM` |

## Results

Holdout accuracy and macro F1, mean over three seeds (42, 43, 44), with the
checkpoint chosen on client validation data. Downlink is the broadcast size per
client and round on CICIoT2023. Standard deviations and per-seed values are in
`runs_v2/<dataset>/summary.json` and in the paper.

| Model | CICIoT2023 Acc. (%) | Macro F1 | NF-ToN-IoT-v2 Acc. (%) | Macro F1 | Downlink (KiB) |
|---|---|---|---|---|---|
| GF-IDS | 97.60 | 0.610 | 91.72 | 0.641 | 23.51 |
| FP32-Twin | 98.68 | 0.688 | 92.88 | 0.795 | 62.26 |
| MLP-FL | 98.07 | 0.648 | 92.78 | 0.799 | 63.76 |
| MLP-INT8 | 88.76 | 0.551 | 92.73 | 0.799 | 18.96 |
| BNN-INT8IO | 86.26 | 0.544 | 89.94 | 0.612 | 8.70 |
| BNN-Full | 71.56 | 0.361 | 73.09 | 0.434 | 4.38 |
| CNN-FL | 85.70 | 0.592 | 93.02 | 0.823 | 333.38 |
| LSTM-FL | 94.67 | 0.600 | 87.88 | 0.618 | 209.51 |

GF-IDS stays about one accuracy point below its full-precision twin and sends
a broadcast about 62% smaller. The upload carries Float32 latent weights and is
not compressed. The energy figures in the paper are an arithmetic estimate, not
a hardware measurement. Under strong label skew on CICIoT2023, GF-IDS loses
clearly more than its twin. The paper lists all limitations.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is optional but speeds up training a lot. One CICIoT2023 run of 45
rounds takes about one to two hours on a single consumer GPU.

## Data

Neither dataset is redistributed.

- **CICIoT2023** from the Canadian Institute for Cybersecurity,
  https://www.unb.ca/cic/datasets/iotdataset-2023.html. Extract it so that
  `./ciciot2023/CICIOT23/` holds the `train/`, `test/` and `validation/` CSV
  folders, or set `GFIDS_CIC_RAW` to another location.
- **NF-ToN-IoT-v2**, https://researchdata.edu.au/nf-ton-iot-v2/3370542. Put
  `NF-ToN-IoT-v2-train.csv` and `NF-ToN-IoT-v2-test.csv` in
  `./raw/NF-ToN-IoT-v2/`, or set `GFIDS_NFTON_RAW`.

## Reproducing the experiments

All commands run from the repository root.

**1. Prepare the training arrays.**

```bash
python code/preprocessing.py
python code/prepare_nfton.py
```

These write the development arrays to `./data/` and `./data_nfton/`.

**2. Prepare the holdouts.**

```bash
python code/prepare_v2.py --dataset cic
python code/prepare_v2.py --dataset nfton
```

This rebuilds the fitted transform, checks it against the arrays of step 1 and
writes the holdout arrays and `transform.json` into `./data_v2/`. The holdout
plays no part in training or in choosing checkpoints.

**3. Train.** One run:

```bash
python code/fl_v2.py --dataset cic --model BNN-MATCHED --seed 42
```

Useful options are `--partition dirichlet --alpha 0.5` for label skew,
`--clients`, `--rounds`, `--epochs` and `--resume`. Each run writes to
`runs_v2/<dataset>/<model>_<setting>_s<seed>_<hash>/`. A finished run is never
overwritten, and an interrupted run continues from its last round with
`--resume`. Every round encodes the broadcast and the upload with
`code/codec.py` and decodes them before use, so the reported message sizes
belong to messages that were actually used.

The queues of the paper, in PowerShell:

```powershell
powershell -File run_v2_queue.ps1 -Dataset nfton
powershell -File run_v2_final.ps1
```

**4. Score the holdout.** Run this only after training has ended.

```bash
python code/eval_v2.py --dataset cic
python code/eval_v2.py --dataset nfton
```

This scores the validation-selected and the last-round checkpoint on the
holdout and on the holdout rows that do not occur in the training pool. It
writes `results.json` for every run and `summary.json` for every dataset.

**5. Tables and figures.**

```bash
python code/make_tables_v2.py --out tables
python code/bold_tables.py --dir tables
python code/make_figs_results_v2.py --out figures_out
```

## Protocol in short

- Five clients, 45 rounds, five local epochs, Adam with learning rate 5e-4,
  batch size 256, FedAvg weighted by client data size.
- Each client keeps a stratified 20% of its share as validation data. The best
  round is the one with the highest pooled validation macro F1.
- Hidden weights are trained with a straight-through estimator on real-valued
  latent weights that are clipped to [-1, 1]. After averaging, the server
  projects the hidden latents to signs and broadcasts them packed, one bit per
  weight.
- The holdout is scored once, after training. The development test set was used
  only in the earlier exploratory runs.

## Exploratory runs

`runs/`, `runs_nfton/` and `runs_unsw/` hold the first runs of the project,
which picked checkpoints on the development test set. The paper uses them only
where it says so, for the aggregation-rule, BiPruneFL and larger-network
comparisons on CICIoT2023. They are kept as a record and are not part of the
reported main results.

## Contact

Please open an issue on this repository.
