"""
Inference benchmark for protocol-v2 checkpoints (PC or Raspberry Pi).

What it measures, and what it does not
  * Latency of the exact PyTorch model that was evaluated for accuracy.
    Binary layers run as ordinary floating point matrix products here,
    so this is NOT a measurement of a packed XNOR kernel.
  * Batch-1 latency (one flow at a time, the gateway case) and
    batch-1024 throughput, each after warm-up, over repeated runs, with
    median and 95th percentile. On CUDA every timing is synchronized.
  * Stored artifact size: the length of the packed downlink message
    (codec.py), i.e. what a gateway has to hold to run the detector.
  * Optional energy (Linux + INA219 only): power is sampled in a
    background thread throughout the timed workload and over an idle
    window of the same length, then integrated (trapezoid rule). Energy
    per inference is reported both total and idle-subtracted. Two
    endpoint readings, as in the v1 script, cannot give energy.

Files needed on the device, in one folder: bench_inference_v2.py,
models.py, binary_ops.py, quant_ops.py, edgepop_ops.py, codec.py, the
sample file (--data) and the run directories or best.pt files.

Usage
    python bench_inference_v2.py --runs runs_v2/nfton --data sample.npz
"""

import argparse
import glob
import json
import os
import platform
import threading
import time

import numpy as np
import torch

from codec import encode
from edgepop_ops import edgepop_frozen_keys
from models import MODEL_REGISTRY


def read_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def environment(device, threads):
    env = {"platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(), "torch": torch.__version__,
           "device": str(device), "torch_threads": threads}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    env["cpu_governor"] = read_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    env["cpu_freq_khz"] = read_file(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return env


def temperature_c():
    t = read_file("/sys/class/thermal/thermal_zone0/temp")
    return int(t) / 1000 if t else None


class INA219Sampler:
    """Samples bus voltage x current at a fixed period in a thread."""

    def __init__(self, period_s=0.002):
        import board
        import busio
        from adafruit_ina219 import INA219
        self.sensor = INA219(busio.I2C(board.SCL, board.SDA))
        self.period = period_s
        self.samples = []
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(),
                                 self.sensor.bus_voltage
                                 * self.sensor.current / 1000.0))   # W
            time.sleep(self.period)

    def measure(self, fn):
        self.samples, self._stop = [], threading.Event()
        th = threading.Thread(target=self._run, daemon=True)
        th.start()
        out = fn()
        self._stop.set()
        th.join()
        t = np.array([s[0] for s in self.samples])
        p = np.array([s[1] for s in self.samples])
        energy = float(np.trapz(p, t)) if len(t) > 1 else None
        duration = float(t[-1] - t[0]) if len(t) > 1 else None
        return out, energy, duration, len(t)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_batches(model, X, bs, n_warm, n_reps, device):
    xs = [torch.from_numpy(X[i:i + bs]).to(device)
          for i in range(0, max(len(X) - bs + 1, 1), bs)] or \
         [torch.from_numpy(X[:bs]).to(device)]
    for i in range(n_warm):
        model(xs[i % len(xs)])
    sync(device)
    times = []
    for i in range(n_reps):
        x = xs[i % len(xs)]
        t0 = time.perf_counter()
        model(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def load_runs(path):
    if path.endswith(".pt"):
        return [(os.path.basename(os.path.dirname(path)) or path, path, None)]
    out = []
    for rd in sorted(glob.glob(os.path.join(path, "*", ""))):
        cfg = os.path.join(rd, "config.json")
        if os.path.exists(os.path.join(rd, "best.pt")) and os.path.exists(cfg):
            c = json.load(open(cfg))["config"]
            out.append((os.path.basename(rd.rstrip("/\\")),
                        os.path.join(rd, "best.pt"), c["model"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="runs_v2/<dataset> directory or one best.pt")
    ap.add_argument("--model", help="model name when --runs is a .pt file")
    ap.add_argument("--data", required=True,
                    help=".npz with X (float32 rows) or an X_*.npy file")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--energy-seconds", type=float, default=0.0,
                    help=">0 enables INA219 energy over a workload of about "
                         "this duration plus an equal idle window")
    ap.add_argument("--seeds", default="42", help="comma list; only these "
                    "seeds are benchmarked when --runs is a directory")
    ap.add_argument("--out", default="bench_inference_v2.json")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    X = (np.load(a.data)["X"] if a.data.endswith(".npz")
         else np.load(a.data, mmap_mode="r")[:20000])
    X = np.ascontiguousarray(X, dtype=np.float32)
    sampler = INA219Sampler() if a.energy_seconds > 0 else None
    keep = {f"_s{s}_" for s in a.seeds.split(",")}

    report = {"environment": environment(device, a.threads), "results": []}
    for run_id, ck, name in load_runs(a.runs):
        name = name or a.model
        if os.path.isdir(a.runs) and not any(k in run_id for k in keep):
            continue
        state = torch.load(ck, map_location="cpu")
        out_key = [k for k in state if k.endswith("weight")][-1]
        C = state[out_key].shape[0]
        model = MODEL_REGISTRY[name](X.shape[1], C)
        model.load_state_dict(state)
        model.eval().to(device)
        excl = edgepop_frozen_keys(model)
        row = {"run_id": run_id, "model": name,
               "artifact_bytes": len(encode(model.cpu(), model.cpu(), "down",
                                            excl)),
               "temp_before_c": temperature_c()}
        model.to(device)
        for bs in (1, 1024):
            t = time_batches(model, X, bs, a.warmup, a.reps if bs == 1
                             else max(a.reps // 20, 50), device)
            row[f"b{bs}_median_ms"] = float(np.median(t) * 1e3)
            row[f"b{bs}_p95_ms"] = float(np.percentile(t, 95) * 1e3)
            row[f"b{bs}_reps"] = int(len(t))
        if sampler:
            n = max(int(a.energy_seconds / (row["b1_median_ms"] / 1e3)), 1)

            def work():
                time_batches(model, X, 1, 0, n, device)
            _, e_load, d_load, k1 = sampler.measure(work)
            _, e_idle, d_idle, k2 = sampler.measure(
                lambda: time.sleep(a.energy_seconds))
            p_idle = e_idle / d_idle
            row.update({"energy_inferences": n, "energy_samples": k1,
                        "workload_s": d_load, "energy_total_J": e_load,
                        "idle_power_W": p_idle,
                        "uJ_per_inference_total": e_load / n * 1e6,
                        "uJ_per_inference_above_idle":
                            (e_load - p_idle * d_load) / n * 1e6})
        row["temp_after_c"] = temperature_c()
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
