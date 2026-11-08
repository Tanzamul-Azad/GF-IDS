# Reproducing the archived experiments

## Environment

```bash
python -m pip install -r requirements.txt
```

Run records include the Python, PyTorch and device details recorded at training
time. A reported CIC run, for example, records Python 3.13.7, PyTorch
2.6.0+cu124 and an NVIDIA GeForce RTX 4060. `requirements.txt` gives minimum
package versions, not a complete historical environment lock. Install a
PyTorch build appropriate for the available platform. A CUDA GPU is optional
but training on CPU can be slow.

## 1. Inspect records without downloading data

```bash
python code/check_run_records.py
```

This uses Python's standard library only. Each run directory contains
`config.json`, `history.json`, `done.json` and `results.json`.
The dataset `summary.json` files contain condition-level aggregates.

## 2. Prepare the exact input files

Follow [DATA.md](DATA.md), including its provenance limitation and raw hashes.
The client split is controlled by the split seed. The transform is fitted once
on the prepared training pool and reused for validation and holdout scoring.

## 3. Train one configuration

```bash
python code/fl_v2.py --dataset cic --model BNN-MATCHED --seed 42
python code/fl_v2.py --dataset nfton --model MLP-TWIN --seed 42
```

For label skew, add `--partition dirichlet --alpha 0.5` or `--alpha 0.1`.
NF controls use `--class-weight sqrt` or `--aggregator latent_broadcast`.
Read the archived configuration for each exact condition.

```bash
python code/fl_v2.py --dataset cic --model BNN-MATCHED --seed 42 --resume
```

Finished run directories are not overwritten. Resume restores saved state;
reproducing a historical run also requires matching code and data. A seed alone
does not guarantee bitwise equality across different libraries or hardware.
Code and data fingerprints in each run record preserve that provenance.

The PowerShell queues are conveniences, not a replacement for reading the
configuration:

```powershell
powershell -File run_v2_queue.ps1 -Dataset nfton
powershell -File run_v2_final.ps1
```

The second command covers the later CIC baseline and non-IID additions. It
does not represent the whole matrix by itself. See [PROTOCOL.md](PROTOCOL.md).

## 4. Evaluate after training

```bash
python code/eval_v2.py --dataset cic
python code/eval_v2.py --dataset nfton
```

Evaluation requires the regenerated checkpoints and arrays. It scores both
validation-selected and final checkpoints on the full and non-overlap
holdouts. Do not use holdout results to choose settings or checkpoints.

## 5. Generate tables and figures

```bash
python code/make_tables_v2.py --out output_tables
python code/bold_tables.py --dir output_tables
python code/make_figs_results_v2.py --out output_figures
python code/make_fig_convergence_v2.py --out output_figures
```

The first script also writes `numbers_v2.json`. The result-figure script
generates per-class and sign-flip plots; it does not regenerate every plot in
the manuscript. Other figure scripts document their own inputs and may need
checkpoints, arrays or cached predictions. Do not substitute exploratory
figure inputs for protocol-v2 results.

## Evidence limits

The archived simulation uses five clients and full participation. There is no
deployed network, device/time-separated split, measured gateway energy or
packed binary inference kernel. Uplink remains Float32. External-method
comparisons in the exploratory records are adaptations rather than faithful
reproductions. Model weights and datasets are excluded from this repository.
