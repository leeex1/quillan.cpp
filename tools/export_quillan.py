#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN.CPP MODEL EXPORTER -- .qbin v1 / v2
=============================================================================
v1 (legacy): magic QLLN (0x4E4C4C51) version 1 + 44-byte config. Kept byte-
             compatible for old PyTorch checkpoints; config-only payload.
v2 (native): magic QLLN version 2 + the SAME 44-byte config prefix, followed by
             an architecture descriptor and a named tensor section table with
             REAL tensor bytes copied straight out of packed safetensors.

Packed safetensors are produced by quillan_sovereign_bitnet.py:
  embed_tokens.weight_int8   I8  [vocab, hidden]
  embed_tokens.weight_scale  F16 [vocab, 1]
  layers.N.*.q_packed        U8  [planes, ceil(out*in/4)]
  layers.N.*.gamma           F16 [planes, out, groups]
  layers.N.*.lora_A/B        F32 [rank, in] / [out, rank]
  layers.N.*.bias            F32 [out] (optional, qkv_bias)
  norm.weight / *.layernorm.weight  F32 [hidden]
  metadata quillan_sovereign_config (JSON) carries the full architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUILLAN_MAGIC = 0x4E4C4C51
QUILLAN_VERSION_V1 = 1
QUILLAN_VERSION_V2 = 2
QLLN_MAX_MOE_LAYERS = 64
QLLN_NAME_LEN = 64
QLLN_ARCH_NAME_LEN = 64
QLLN_CHUNK_BYTES = 16 * 1024 * 1024

# quillan_config = 9 * int32 + 2 * float = 44 bytes
CFG_FMT = "<9i2f"
# arch descriptor = 10i + 64i + 2f + i + 64s = 372 bytes
ARCH_FMT = "<10i64i2fi64s"
# tensor section = 64s + 2*uint32 + 4*int32 + 2*uint64 = 104 bytes
SEC_FMT = "<64sII4iQQ"

DT_SIZE = {"F32": 4, "F16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 4}
DT_CODE = {"F32": 0, "F16": 1, "I8": 2, "U8": 3, "I32": 4, "I64": 4}

ARCH_ID = {
    "quillan-mini-bitnet": 1,
    "quillan-flagship-bitnet": 2,
    "quillan-dragon-bitnet": 3,
}


def log(msg: str) -> None:
    print(f"[export_quillan] {msg}")


def read_metadata(cfg_path: Path) -> Optional[Dict[str, Any]]:
    """Reads the sovereign architecture JSON from a packed safetensors header."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    with safe_open(str(cfg_path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        raw = meta.get("quillan_sovereign_config")
        return json.loads(raw) if raw else None


def is_packed(cfg: Optional[Dict[str, Any]], keys: List[str]) -> bool:
    if not cfg:
        return False
    required = ("hidden_size", "vocab_size", "num_hidden_layers", "head_dim")
    if not all(k in cfg for k in required):
        return False
    return "embed_tokens.weight_int8" in keys and any(
        k.endswith(".q_packed") for k in keys
    )


def build_v2_header(
    cfg: Dict[str, Any],
    entries: List[Tuple[str, int, int, Tuple[int, ...], int]],
    max_seq_len: int,
    arch_override: Optional[int],
    name_override: Optional[str],
) -> Tuple[bytes, int]:
    vocab = int(cfg["vocab_size"])
    hidden = int(cfg["hidden_size"])
    ffn = int(cfg["intermediate_size"])
    layers = int(cfg["num_hidden_layers"])
    experts = int(cfg.get("num_experts", 34))
    active = int(cfg.get("num_experts_active", 2))
    rank = int(cfg.get("eggroll_rank", 16))
    orig_pos = int(cfg.get("orig_max_position_embeddings", 32768))
    max_seq = max(1, min(int(max_seq_len), orig_pos))

    base_cfg = struct.pack(
        CFG_FMT,
        vocab,
        hidden,
        ffn,
        layers,
        experts,
        active,
        rank,
        max_seq,
        0,  # diffusion_steps: 0 for the real transformer
        0.92,
        0.85,  # diffusion_halt / ccrl_threshold
    )

    moe_indices = [int(x) for x in cfg.get("moe_layer_indices", [])]
    if len(moe_indices) > QLLN_MAX_MOE_LAYERS:
        raise ValueError(
            f"too many MoE layers: {len(moe_indices)} > {QLLN_MAX_MOE_LAYERS}"
        )
    moe_indices += [-1] * (QLLN_MAX_MOE_LAYERS - len(moe_indices))

    model_name = str(cfg.get("model_name", "custom"))
    arch_id = (
        arch_override
        if arch_override is not None
        else ARCH_ID.get(model_name, 0x7FFFFFFF)
    )
    arch_name = (name_override or model_name).encode("utf-8")[: QLLN_ARCH_NAME_LEN - 1]

    flags = 0
    if cfg.get("qkv_bias", False):
        flags |= 1
    if cfg.get("tie_word_embeddings", True):
        flags |= 2
    if cfg.get("use_rht", True):
        flags |= 4

    arch = struct.pack(
        ARCH_FMT,
        int(arch_id),
        int(cfg["head_dim"]),
        int(cfg["num_attention_heads"]),
        int(cfg["num_key_value_heads"]),
        len([m for m in moe_indices if m >= 0]),
        int(cfg.get("attn_planes", 3)),
        int(cfg.get("ffn_planes", 3)),
        int(cfg.get("group_size", 32)),
        1,  # packed_format = PACKED_TERNARY_V1
        flags,
        *moe_indices,
        float(cfg.get("rms_norm_eps", 1e-6)),
        float(cfg.get("rope_theta", 1e6)),
        orig_pos,
        arch_name,
    )
    assert len(base_cfg) == 44 and len(arch) == 372

    header = struct.pack("<II", QUILLAN_MAGIC, QUILLAN_VERSION_V2) + base_cfg + arch
    header += struct.pack("<I", len(entries))
    offset = 0
    for name, dtype, rank, dims, nbytes in entries:
        dims4 = tuple(dims) + (0,) * (4 - rank)
        header += struct.pack(
            SEC_FMT,
            name.encode("utf-8")[: QLLN_NAME_LEN - 1],
            dtype,
            rank,
            *dims4,
            offset,
            nbytes,
        )
        offset += nbytes
    return header, offset


def stream_tensor(f: Any, name: str, shape: Tuple[int, ...], out: Any) -> int:
    """Copies raw tensor bytes in bounded chunks; returns the byte count."""
    if not shape:
        data = f.get_tensor(name).contiguous().numpy().tobytes()
        out.write(data)
        return len(data)
    row_elems = math.prod(shape[1:])
    rows_per_chunk = max(1, QLLN_CHUNK_BYTES // max(1, row_elems * 4))
    sl = f.get_slice(name)
    written = 0
    start = 0
    while start < shape[0]:
        end = min(shape[0], start + rows_per_chunk)
        part = sl[start:end].contiguous().numpy().tobytes()
        out.write(part)
        written += len(part)
        start = end
    return written


def export_packed(
    ckpt: Path,
    out_path: Path,
    cfg: Dict[str, Any],
    max_seq_len: int,
    arch_override: Optional[int],
    name_override: Optional[str],
) -> None:
    from safetensors import safe_open

    log(f"Detected PACKED safetensors -> v2 export")
    log(
        f"  arch={cfg.get('model_name')} vocab={cfg['vocab_size']} hidden={cfg['hidden_size']} "
        f"layers={cfg['num_hidden_layers']} heads={cfg['num_attention_heads']} "
        f"head_dim={cfg['head_dim']} kv_heads={cfg['num_key_value_heads']}"
    )

    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        entries: List[Tuple[str, int, int, Tuple[int, ...], int]] = []
        for name in f.keys():
            sl = f.get_slice(name)
            shape = tuple(int(x) for x in sl.get_shape())
            dtype = str(sl.get_dtype())
            if dtype not in DT_CODE:
                raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
            nbytes = math.prod(shape) * DT_SIZE[dtype]
            entries.append((name, DT_CODE[dtype], len(shape), shape, nbytes))

        header, expected_payload = build_v2_header(
            cfg, entries, max_seq_len, arch_override, name_override
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as out:
            out.write(header)
            written = 0
            for name, _dt, _rank, shape, _nb in entries:
                written += stream_tensor(f, name, shape, out)

    total = out_path.stat().st_size
    if written != expected_payload:
        raise IOError(
            f"payload size mismatch: wrote {written}, expected {expected_payload}"
        )
    log(
        f"Wrote {len(entries)} real tensor sections ({written / (1024 * 1024):.1f} MB payload)"
    )
    log(f"Output: {out_path} ({total / (1024 * 1024):.1f} MB)")
    verify_v2(out_path)


def export_legacy(
    pt_path: Path, out_path: Path, hidden_dim: int, ffn_dim: int, num_layers: int
) -> None:
    """Preserves the original v1 config-only export for unpacked .pt checkpoints."""
    import torch

    log(f"Legacy checkpoint -> v1 header-only export (no packed tensor sections)")
    state: Dict[str, Any] = {}
    if pt_path.exists():
        if pt_path.suffix == ".safetensors":
            log(
                "WARNING: safetensors without sovereign metadata; v1 config-only payload"
            )
        else:
            loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                state = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
    detected = set()
    for k in state:
        for marker in ("layers.", "h."):
            if marker in k:
                parts = k.split(marker)[1].split(".")
                if parts and parts[0].isdigit():
                    detected.add(int(parts[0]))
    if detected:
        num_layers = len(detected)
    header = struct.pack(
        "<II" + CFG_FMT.lstrip("<>=!@"),
        QUILLAN_MAGIC,
        QUILLAN_VERSION_V1,
        50257,
        hidden_dim,
        ffn_dim,
        num_layers,
        34,
        4,
        8,
        512,
        3,
        0.92,
        0.85,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(bytes(1024))
    log(f"Output: {out_path} ({out_path.stat().st_size} bytes, layers={num_layers})")


def verify_v2(path: Path) -> None:
    with open(path, "rb") as f:
        head = f.read(8 + 44 + 372 + 4)
    if len(head) < 428:
        raise IOError("v2 file truncated")
    magic, version = struct.unpack_from("<II", head, 0)
    cfg = struct.unpack_from(CFG_FMT, head, 8)
    arch = struct.unpack_from(ARCH_FMT, head, 52)
    count = struct.unpack_from("<I", head, 424)[0]
    expected = 428 + count * 104
    actual = path.stat().st_size
    log(
        f"VERIFY magic=0x{magic:08X} v{version} sections={count} "
        f"arch_id={arch[0]} head_dim={arch[1]} heads={arch[2]} kv={arch[3]} "
        f"MoE_layers={arch[4]} packed_fmt={arch[8]} size_ok={actual >= expected}"
    )
    assert (
        magic == QUILLAN_MAGIC and version == QUILLAN_VERSION_V2 and actual >= expected
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export Quillan checkpoints to .qbin v1/v2 for quillan.cpp"
    )
    ap.add_argument(
        "--checkpoint",
        default=str(
            Path("C:/02_QUILLAN/exports/quillan-mini-sovereign-bitnet.safetensors")
        ),
    )
    ap.add_argument("--out", default="quillan_v5_4.qbin")
    ap.add_argument("--hidden-dim", type=int, default=1024, help="v1 legacy only")
    ap.add_argument("--num-layers", type=int, default=6, help="v1 legacy only")
    ap.add_argument(
        "--max-seq-len", type=int, default=2048, help="v2 KV-cache context cap"
    )
    ap.add_argument(
        "--arch-id", type=int, default=None, help="override v2 architecture id"
    )
    ap.add_argument("--arch-name", default=None, help="override v2 architecture name")
    ap.add_argument("--legacy", action="store_true", help="force v1 export")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    if not ckpt.is_file():
        log(f"ERROR checkpoint not found: {ckpt}")
        sys.exit(2)

    cfg: Optional[Dict[str, Any]] = None
    keys: List[str] = []
    if not args.legacy and ckpt.suffix == ".safetensors":
        cfg = read_metadata(ckpt)
        from safetensors import safe_open

        with safe_open(str(ckpt), framework="pt", device="cpu") as f:
            keys = list(f.keys())

    if not args.legacy and is_packed(cfg, keys):
        export_packed(ckpt, out, cfg, args.max_seq_len, args.arch_id, args.arch_name)
    else:
        export_legacy(ckpt, out, args.hidden_dim, 2048, args.num_layers)


if __name__ == "__main__":
    main()
