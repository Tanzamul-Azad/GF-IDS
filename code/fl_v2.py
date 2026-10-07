"""
Federated training under the corrected (v2) protocol.

What changed from federated_train.py (v1), and why
  * No test data is loaded. Checkpoints are chosen on the clients'
    held-out 20% (pooled validation), by validation macro F1. The
    final holdout is scored only by eval_v2.py, after training ends.
  * Every round really encodes the downlink and uplink messages with
    codec.py and decodes them before use, so the reported bytes belong
    to messages that were used, not to a counting helper.
  * The initial broadcast is explicit. The default ("float") matches
    v1: round 1 starts from the real-valued initialization, sent once
    in Float32 and counted separately. "projected" sends signs from the
    start; latent weights then sit at +-1, the edge of the clip window,
    and a hidden weight can flip only after a cumulative move of 1.
  * BatchNorm running mean/variance are sample-weighted averages. The
    integer num_batches_tracked counter is not averaged or sent.
  * Batches are drawn on the GPU from a seeded torch.Generator whose
    state is checkpointed with the CPU/CUDA RNG states, so a resumed
    run replays an uninterrupted one. v1 restarted shuffling from a
    fresh RNG on resume.
  * Each run lives in its own directory named by model, partition,
    seed and a hash of the full configuration. A finished run is never
    overwritten, and a resume under different code is refused.

Usage (from the repository root)
    python code/fl_v2.py --dataset nfton --model BNN-MATCHED --seed 42
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from binary_ops import binary_weight_keys, clip_all_binary_weights
from codec import decode, encode
from edgepop_ops import edgepop_frozen_keys
from metrics_v2 import benign_index
from models import MODEL_REGISTRY
from quant_ops import fake_quant, quant_weight_keys

DATA_DIRS = {"cic": "./data/", "nfton": "./data_nfton/"}
OUT_ROOT = os.environ.get("GFIDS_V2_OUT", "./runs_v2/")
CODE_FILES = ["fl_v2.py", "codec.py", "models.py", "binary_ops.py",
              "quant_ops.py", "edgepop_ops.py", "metrics_v2.py"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── provenance ──────────────────────────────────────────────
def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def code_fingerprint():
    here = os.path.dirname(os.path.abspath(__file__))
    return {f: file_sha256(os.path.join(here, f)) for f in CODE_FILES}


def data_fingerprint(ddir):
    """SHA-256 of the training arrays, cached by size and mtime."""
    cache = os.path.join(OUT_ROOT, "data_hashes.json")
    known = json.load(open(cache)) if os.path.exists(cache) else {}
    out = {}
    for f in ("X_train.npy", "y_train.npy"):
        p = os.path.abspath(os.path.join(ddir, f))
        st = os.stat(p)
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key not in known:
            known[key] = file_sha256(p)
        out[f] = known[key]
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(known, fh, indent=1)
    os.replace(tmp, cache)
    return out


# ── data ────────────────────────────────────────────────────
def client_splits(y, K, partition, alpha, split_seed, val_fraction):
    """Client index sets. With split_seed=42 and partition='iid' this
    reproduces the v1 seed-42 partition and local 80/20 split exactly."""
    rng = np.random.RandomState(split_seed)
    if partition == "iid":
        perm = rng.permutation(len(y))
        size = len(perm) // K
        shards = [perm[i * size:(i + 1) * size] for i in range(K)]
    else:
        C = int(y.max()) + 1
        shards = [[] for _ in range(K)]
        for c in range(C):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, K))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].extend(part.tolist())
        for k in range(K):
            rng.shuffle(shards[k])
        shards = [np.array(s) for s in shards]
    out = []
    for s in shards:
        ys = y[s]
        _, cnt = np.unique(ys, return_counts=True)
        tr, va = train_test_split(
            s, test_size=val_fraction, random_state=split_seed,
            stratify=ys if cnt.min() >= 2 else None)
        out.append((np.sort(tr), np.sort(va)))
    return out


# ── training pieces ─────────────────────────────────────────
def project(state, binary_keys, int8_keys):
    """Map the server state onto what is broadcast: signs for binary
    weights (q(0) = +1), the int8 grid for quantized weights."""
    for k in binary_keys:
        state[k] = torch.where(state[k] >= 0, 1.0, -1.0).to(state[k].dtype)
    for k, bits in int8_keys.items():
        state[k] = fake_quant(state[k], bits)
    return state


def aggregate(states, sizes, template):
    total = float(sum(sizes))
    out = {}
    for k, ref in template.items():
        if ref.dtype.is_floating_point:
            out[k] = sum(s[k].to(device) * (n / total)
                         for s, n in zip(states, sizes))
        else:
            out[k] = ref.clone()      # integer counters are not averaged
    return out


def local_train(model, X, y, idx, epochs, bs, opt, crit, gen,
                prox_mu=0.0, anchor=None):
    model.train()
    n = len(idx)
    for _ in range(epochs):
        order = idx[torch.randperm(n, generator=gen, device=device)]
        for s in range(0, n, bs):
            b = order[s:s + bs]
            if len(b) < 2:            # BatchNorm needs two samples
                continue
            opt.zero_grad(set_to_none=True)
            loss = crit(model(X[b]), y[b])
            if prox_mu > 0:
                loss = loss + 0.5 * prox_mu * sum(
                    ((p - a) ** 2).sum()
                    for p, a in zip(model.parameters(), anchor))
            loss.backward()
            opt.step()
            clip_all_binary_weights(model)


@torch.no_grad()
def validate(model, X, y, idx, crit, C, b_idx, bs=16384):
    model.eval()
    preds, loss = [], 0.0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        z = model(X[b])
        loss += crit(z, y[b]).item() * len(b)
        preds.append(z.argmax(1))
    p = torch.cat(preds).cpu().numpy()
    t = y[idx].cpu().numpy()
    if device.type == "cuda":
        # Return cached activation memory, so other GPU jobs can run
        # next to training (the LSTM otherwise keeps several GB cached).
        torch.cuda.empty_cache()
    benign = t == b_idx
    return {"val_accuracy": float((p == t).mean()),
            "val_macro_f1": float(f1_score(t, p, labels=np.arange(C),
                                           average="macro", zero_division=0)),
            "val_loss": loss / len(idx),
            "val_benign_far": float((benign & (p != b_idx)).sum()
                                    / max(benign.sum(), 1))}


def class_weights(y, idx_list, C, mode):
    if mode == "none":
        return None
    cnt = np.bincount(np.concatenate([y[i] for i in idx_list]),
                      minlength=C).astype(np.float64)
    w = 1.0 / np.maximum(cnt, 1)
    if mode == "sqrt":
        w = np.sqrt(w)
    w = w * C / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


# ── crash safety ────────────────────────────────────────────
def atomic_save(obj, path):
    """Write to a temporary file, keep the previous copy as .bak, then
    swap. A crash during the write leaves the old file intact."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def acquire_lock(rdir):
    """Exclusive lock held for the life of the process. The operating
    system drops it when the process dies, so a stale lock never blocks
    a resume, while a second live process on the same run is refused."""
    f = open(os.path.join(rdir, "run.lock"), "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"{rdir} is being trained by another process")
    return f


def load_state(path):
    """Latest checkpoint, or the previous one if the latest is damaged."""
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:     # truncated or corrupt file
                print(f"could not read {p}: {e}")
    return None


# Arguments that control how a run is executed, not what it computes.
# They are left out of the configuration hash, so adding or changing one
# never renames an existing run directory.
RUNTIME_ARGS = ("resume", "force_resume", "stop_after", "strict_code")


# ── main loop ───────────────────────────────────────────────
def run(a):
    ddir = DATA_DIRS[a.dataset]
    cfg = {k: v for k, v in vars(a).items() if k not in RUNTIME_ARGS}
    cfg["split_seed"] = a.seed if a.split_seed is None else a.split_seed
    cfg["data_sha256"] = data_fingerprint(ddir)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True)
                              .encode()).hexdigest()[:10]
    part = "iid" if a.partition == "iid" else f"dir{a.alpha}"
    run_id = f"{a.model}_{part}_s{a.seed}_{cfg_hash}"
    rdir = os.path.join(OUT_ROOT, a.dataset, run_id)
    if os.path.exists(os.path.join(rdir, "done.json")):
        print(f"{run_id} already finished; not overwriting")
        return rdir
    os.makedirs(rdir, exist_ok=True)
    lock = acquire_lock(rdir)          # noqa: F841 (held until exit)
    code = code_fingerprint()

    X_np = np.load(os.path.join(ddir, "X_train.npy")).astype(np.float32)
    y_np = np.load(os.path.join(ddir, "y_train.npy")).astype(np.int64)
    names = list(np.load(os.path.join(ddir, "class_names.npy"),
                         allow_pickle=True))
    C, d = len(names), X_np.shape[1]
    b_idx = benign_index(names)
    splits = client_splits(y_np, a.clients, a.partition, a.alpha,
                           cfg["split_seed"], a.val_fraction)
    tr_sizes = [len(t) for t, _ in splits]
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    del X_np
    tr_idx = [torch.from_numpy(t).to(device) for t, _ in splits]
    va_all = torch.from_numpy(np.concatenate([v for _, v in splits])).to(device)

    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed)
    Model = MODEL_REGISTRY[a.model]
    gmodel = Model(d, C).to(device)
    bkeys = binary_weight_keys(gmodel)
    qkeys = quant_weight_keys(gmodel)
    excl = edgepop_frozen_keys(gmodel)
    w = class_weights(y_np, [t for t, _ in splits], C, a.class_weight)
    crit = nn.CrossEntropyLoss(weight=w)
    vcrit = nn.CrossEntropyLoss()     # validation loss is unweighted

    history, start = [], 0
    best = {"round": 0, "val_macro_f1": -1.0}
    state_path = os.path.join(rdir, "state.pt")
    # RNG states must stay on the CPU, so the state is loaded there.
    st = load_state(state_path) if a.resume else None
    if st is not None:
        if st["code"] != code:
            # Resuming under edited code is allowed but never silent: the
            # change is logged in the run directory so it stays auditable.
            if a.strict_code:
                raise SystemExit("code changed since this run started; "
                                 "refusing to resume (--strict-code)")
            log = os.path.join(rdir, "code_changes.json")
            entries = json.load(open(log)) if os.path.exists(log) else []
            entries.append({"resumed_at_round": st["round"] + 1,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "changed_files": sorted(
                                f for f in code
                                if code[f] != st["code"].get(f)),
                            "previous": st["code"], "current": code})
            atomic_json(entries, log)
            print(f"note: code changed since round {st['round']}; "
                  f"logged in {log}")
        gmodel.load_state_dict(st["model"])
        history, start, best = st["history"], st["round"], st["best"]
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        mem = {k: v.to(device) for k, v in st.get("mem", {}).items()}
        print(f"resuming {run_id} at round {start + 1}")
    else:
        if a.init_broadcast == "projected":
            gmodel.load_state_dict(project(gmodel.state_dict(), bkeys, qkeys))
        init_dir = "down" if a.init_broadcast == "projected" else "up"
        atomic_json({"run_id": run_id, "config": cfg, "code_sha256": code,
                   "torch": torch.__version__, "python": platform.python_version(),
                   "device": torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu",
                   "client_train_sizes": tr_sizes,
                   "client_val_sizes": [len(v) for _, v in splits],
                   "client_classes_present": [int(len(np.unique(y_np[t])))
                                              for t, _ in splits],
                   "initial_broadcast_bytes": len(encode(gmodel, gmodel,
                                                         init_dir, excl)),
                   "selection_rule": "max pooled validation macro F1; "
                                     "earliest round on ties"},
                    os.path.join(rdir, "config.json"))
        # Server latent memory (aggregator "latent_memory"): the real-valued
        # average of the hidden latents, kept across rounds. Starts at the
        # initialization that round 1 broadcasts.
        mem = ({k: gmodel.state_dict()[k].clone() for k in bkeys}
               if a.aggregator == "latent_memory" else {})

    def signs(m):
        return {k: m.state_dict()[k] >= 0 for k in bkeys}
    prev_signs = signs(gmodel) if bkeys else None
    for t in range(start, a.rounds):
        t0 = time.time()
        lr = max(a.lr_min, a.lr * a.lr_decay ** t)
        # Round 1 under "float" init receives the raw initialization.
        ddir_ = "up" if ((t == 0 and a.init_broadcast == "float")
                         or a.aggregator == "latent_broadcast") else "down"
        down = encode(gmodel, gmodel, ddir_, excl)
        recv = decode(down, gmodel, ddir_, excl)
        down_len = len(down)
        if a.aggregator == "latent_memory" and ddir_ == "down":
            # One Float32 magnitude per binary layer, alpha = mean |memory|.
            # Clients start from alpha * signs, inside the clip window,
            # so their change is not forced toward zero by the clip.
            for k_ in bkeys:
                recv[k_] = recv[k_] * mem[k_].abs().mean().cpu()
            down_len += 4 * len(bkeys)
        ups, up_bytes = [], []
        for k in range(a.clients):
            local = Model(d, C).to(device)
            local.load_state_dict(recv)
            opt = torch.optim.Adam(local.parameters(), lr=lr,
                                   fused=device.type == "cuda")
            anchor = ([p.detach().clone() for p in local.parameters()]
                      if a.aggregator == "fedprox" else None)
            local_train(local, X, y, tr_idx[k], a.epochs, a.batch_size, opt,
                        crit, gen, a.mu if a.aggregator == "fedprox" else 0.0,
                        anchor)
            msg = encode(local, local, "up", excl)
            up_bytes.append(len(msg))
            ups.append(decode(msg, gmodel, "up", excl))
        new = aggregate(ups, tr_sizes, gmodel.state_dict())
        if a.aggregator == "latent_memory":
            # Clients started from the broadcast signs (or the round-1
            # initialization). Their weighted mean change is added to the
            # server memory, which is clipped like a client latent; the
            # broadcast stays the projected signs of the memory.
            for k_ in bkeys:
                mem[k_] = (mem[k_] + a.server_lr * (new[k_] - recv[k_].to(device))
                           ).clamp(-1, 1)
                new[k_] = mem[k_].clone()
            gmodel.load_state_dict(project(new, bkeys, qkeys))
        elif a.aggregator == "latent_broadcast":
            # Control: no sign projection; latents travel in Float32.
            gmodel.load_state_dict(project(new, set(), qkeys))
        else:
            gmodel.load_state_dict(project(new, bkeys, qkeys))

        flip = None
        if bkeys:
            cur = signs(gmodel)
            flip = (sum((cur[k] != prev_signs[k]).sum().item() for k in bkeys)
                    / sum(cur[k].numel() for k in bkeys))
            prev_signs = cur
        v = validate(gmodel, X, y, va_all, vcrit, C, b_idx)
        rec = {"round": t + 1, "lr": lr, "down_bytes": down_len,
               "up_bytes_per_client": up_bytes, "sign_flip_rate": flip,
               "seconds": round(time.time() - t0, 2), **v}
        history.append(rec)
        if v["val_macro_f1"] > best["val_macro_f1"]:
            best = {"round": t + 1, **v}
            atomic_save(gmodel.state_dict(), os.path.join(rdir, "best.pt"))
        atomic_save({"model": gmodel.state_dict(), "history": history,
                    "round": t + 1, "best": best, "gen": gen.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "mem": {k: v.cpu() for k, v in mem.items()},
                    "code": code}, state_path)
        print(f"{run_id} r{t + 1:2d}/{a.rounds} valF1={v['val_macro_f1']:.4f} "
              f"valAcc={v['val_accuracy']*100:.2f} "
              f"valFAR={v['val_benign_far']*100:.2f} "
              f"down={down_len}B {rec['seconds']}s"
              + (f" flip={flip:.5f}" if flip is not None else ""), flush=True)
        if a.stop_after and t + 1 >= a.stop_after and t + 1 < a.rounds:
            print(f"stopping after round {t + 1} (--stop-after)")
            return rdir

    atomic_save(gmodel.state_dict(), os.path.join(rdir, "final.pt"))
    atomic_json(history, os.path.join(rdir, "history.json"))
    # done.json is written last: it is the marker that the run finished.
    atomic_json({"best": best, "rounds": len(history),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
                os.path.join(rdir, "done.json"))
    print(f"{run_id} done; best validation round {best['round']} "
          f"(macro F1 {best['val_macro_f1']:.4f})")
    return rdir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATA_DIRS))
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--seed", type=int, default=42,
                   help="weight initialization and batch order")
    p.add_argument("--split-seed", type=int, default=None,
                   help="client partition and local validation split "
                        "(defaults to --seed, so methods with the same seed "
                        "share partitions)")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=45)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--lr-decay", type=float, default=1.0)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--partition", default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--aggregator", default="fedavg",
                   choices=["fedavg", "fedprox", "latent_memory",
                            "latent_broadcast"])
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--server-lr", type=float, default=1.0,
                   help="step applied to the mean client change in the "
                        "latent_memory aggregator")
    p.add_argument("--init-broadcast", default="float",
                   choices=["float", "projected"])
    p.add_argument("--class-weight", default="none",
                   choices=["none", "inv", "sqrt"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-resume", action="store_true",
                   help="kept for compatibility; resuming under changed "
                        "code is now allowed and logged")
    p.add_argument("--strict-code", action="store_true",
                   help="refuse to resume if the code changed")
    p.add_argument("--stop-after", type=int, default=0,
                   help="exit after this round without finishing; used to "
                        "test exact resume")
    a = p.parse_args()
    if a.partition == "iid":
        a.alpha = None
    run(a)


if __name__ == "__main__":
    main()
