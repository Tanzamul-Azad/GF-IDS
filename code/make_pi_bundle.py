"""Collect everything bench_inference_v2.py needs to run on a Raspberry Pi 4.

    python code/make_pi_bundle.py --out ../pi_bundle

Copies the benchmark script, the model/codec modules, the 500-row sample and
the seed-42 IID CICIoT2023 run directories (config.json + best.pt only).
Safe to rerun: models that finished training since the last call are added.
On the Pi (64-bit OS, torch installed), latency only:

    python3 bench_inference_v2.py --runs runs_v2/cic --data pi_bench_sample.npz \
        --threads 1 --out pi_latency.json
"""
import argparse
import glob
import os
import shutil

CODE = ["bench_inference_v2.py", "models.py", "binary_ops.py", "quant_ops.py",
        "edgepop_ops.py", "codec.py", "pi_bench_sample.npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../pi_bundle")
    ap.add_argument("--seed", default="42")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.makedirs(a.out, exist_ok=True)
    for f in CODE:
        shutil.copy2(os.path.join(here, f), a.out)
    n = 0
    for rd in sorted(glob.glob(os.path.join(root, "runs_v2", "cic",
                                            f"*_iid_s{a.seed}_*"))):
        if not os.path.exists(os.path.join(rd, "done.json")):
            continue
        dst = os.path.join(a.out, "runs_v2", "cic", os.path.basename(rd))
        os.makedirs(dst, exist_ok=True)
        for f in ("config.json", "best.pt"):
            shutil.copy2(os.path.join(rd, f), dst)
        n += 1
    print(f"bundle at {os.path.abspath(a.out)}: {n} finished runs")


if __name__ == "__main__":
    main()
