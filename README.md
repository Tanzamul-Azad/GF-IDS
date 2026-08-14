# GF-IDS

Code for simulating federated training of intrusion-detection models on IoT
network traffic, including a variant with binary hidden-layer weights, and for
measuring each model's accuracy and communication cost.

The accompanying paper is under review. Its text and figures are not part of
this repository.

## Repository layout

```
code/
  models.py            model definitions
  preprocessing.py     data cleaning, feature pruning, scaling, encoding
  federated_train.py   federated training simulation
  evaluate.py          detection and efficiency metrics
  figures.py           plotting scripts
```

## Setup

```bash
pip install -r requirements.txt
```

The CICIoT2023 dataset is not redistributed here. Download it from the
Canadian Institute for Cybersecurity and extract it so that
`./ciciot2023/CICIOT23/` contains the `train/` and `test/` CSV folders:

https://www.unb.ca/cic/datasets/iotdataset-2023.html

## Running the code

```bash
python code/preprocessing.py
```
Prepares the raw CSV files and writes `.npy` arrays into `./data/`.

```bash
python code/federated_train.py --model BNN
python code/federated_train.py --model MLP
python code/federated_train.py --model CNN
python code/federated_train.py --model LSTM
```
Runs the federated simulation and writes checkpoints and a per-round history
into `./runs/`. Default settings are the argument defaults in
`code/federated_train.py`.

```bash
python code/evaluate.py
python code/figures.py
```
Computes the metric tables and draws the plots.

## Reading the efficiency numbers

`evaluate.py` reports two payload figures that answer different questions:

- **`IdealPayload(KB)`** counts each binary hidden weight as 1 bit and every
  other value as 32 bits, which describes an idealized 1-bit encoding.
- **`MeasuredPayload(KB)`** is the size of the tensors the training loop
  actually sends, which are full float32 state dicts.

State which one a claim uses. The same care applies to `FLOPs(M)`: `thop`
matches modules by exact type, so `BinaryLinear` (a subclass of `nn.Linear`)
counts as zero unless a custom handler is registered. `evaluate.py` runs
`thop` with `verbose=True`, so check its output before quoting FLOPs.

## Contact

Please open an issue on this repository.
