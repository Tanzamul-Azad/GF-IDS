"""
Model-message codec for GF-IDS (v2).

v1 counted payload bytes with helper functions but never produced a
message that could be decoded and used. Its INT8 path also cast float
weights straight to int8 (w.astype(np.int8)), which truncates almost
every weight to zero. The byte count was right; the message was not.

This module encodes a state dict into one bytes object and decodes it
back. Both ends share the model schema (tensor order, shapes, dtypes),
so no names or shapes travel. What does travel:

  binary tensor   ceil(N / 8) bytes, bit = 1 for w >= 0 (q(0) = +1, as
                  in binary_ops.SignSTE)
  int8 tensor     4-byte Float32 scale + N int8 codes,
                  q = clip(round(w / s), -127, 127), w_hat = s * q
  other tensor    raw little-endian bytes at native dtype

BatchNorm `num_batches_tracked` is excluded. It is an integer counter
that BatchNorm1d ignores when momentum is set (the default here), so it
need not cross the network. Byte counts are tensor payload only. They
exclude transport headers, framing and encryption.
"""

import numpy as np
import torch

from binary_ops import binary_weight_keys
from quant_ops import quant_weight_keys

SKIP_SUFFIXES = ("num_batches_tracked",)


def schema(model, exclude=frozenset()):
    """Tensor order and per-tensor codec for one model class.

    `exclude` drops named tensors that never change after the first
    broadcast (BiPruneFL-Repro's frozen random weights)."""
    binary = binary_weight_keys(model)
    int8 = set(quant_weight_keys(model))
    out = []
    for name, t in model.state_dict().items():
        if name.endswith(SKIP_SUFFIXES) or name in exclude:
            continue
        kind = "raw"
        out.append((name, tuple(t.shape), t.dtype, kind))
    return out, binary, int8


def int8_encode(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.float32(max(np.abs(w).max(), 1e-8) / qmax)
    q = np.clip(np.round(w / s), -qmax, qmax).astype(np.int8)
    return s, q


def encode(model_or_state, model, direction="down", exclude=frozenset()):
    """Encode a state dict.

    direction="down": binary layers packed to 1 bit, int8 layers to
    8 bits, as in the re-binarized / re-quantized global model.
    direction="up": every tensor raw, the latent-weight upload used by
    the current GF-IDS protocol.
    """
    state = (model_or_state.state_dict()
             if hasattr(model_or_state, "state_dict") else model_or_state)
    order, binary, int8 = schema(model, exclude)
    chunks = []
    for name, _, _, _ in order:
        w = state[name].detach().cpu().numpy()
        if direction == "down" and name in binary:
            chunks.append(np.packbits((w >= 0).astype(np.uint8).ravel())
                          .tobytes())
        elif direction == "down" and name in int8:
            s, q = int8_encode(w.astype(np.float32))
            chunks.append(np.array([s], dtype="<f4").tobytes())
            chunks.append(q.tobytes())
        else:
            chunks.append(np.ascontiguousarray(w).astype(w.dtype.newbyteorder("<"))
                          .tobytes())
    return b"".join(chunks)


def decode(msg, model, direction="down", exclude=frozenset()):
    """Inverse of encode(). Returns a state dict that loads into `model`
    (num_batches_tracked is taken from the receiver's own copy)."""
    order, binary, int8 = schema(model, exclude)
    ref = model.state_dict()
    out = {}
    pos = 0
    for name, shape, dtype, _ in order:
        n = int(np.prod(shape)) if shape else 1
        if direction == "down" and name in binary:
            nb = (n + 7) // 8
            bits = np.unpackbits(np.frombuffer(msg, np.uint8, nb, pos))[:n]
            w = bits.astype(np.float32) * 2 - 1
            pos += nb
        elif direction == "down" and name in int8:
            s = np.frombuffer(msg, "<f4", 1, pos)[0]
            pos += 4
            q = np.frombuffer(msg, np.int8, n, pos)
            pos += n
            w = q.astype(np.float32) * s
        else:
            np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
            nbytes = n * np_dtype.itemsize
            w = np.frombuffer(msg, np_dtype.newbyteorder("<"), n, pos).copy()
            pos += nbytes
        out[name] = torch.from_numpy(np.array(w).reshape(shape)).to(
            dtype=dtype)
    if pos != len(msg):
        raise ValueError(f"decoded {pos} of {len(msg)} bytes")
    for name, t in ref.items():
        if name not in out:
            out[name] = t.detach().cpu().clone()
    return out


def message_bytes(model, direction, exclude=frozenset()):
    return len(encode(model, model, direction, exclude))
