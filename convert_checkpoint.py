#!/usr/bin/env python3
"""
Convert a nanoGPT `ckpt.pt` checkpoint into a browser-loadable bundle.

The output is a single binary file with:
- 8-byte magic header
- 4-byte little-endian manifest length
- UTF-8 JSON manifest
- raw float32 tensor payload

This avoids depending on PyTorch pickle loading in the browser.
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Any

import torch

MAGIC = b"NGVIZ01\n"
MANIFEST_DTYPE = "float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to `ckpt.pt`.")
    parser.add_argument(
        "--meta",
        help="Optional path to `meta.pkl`. If omitted, the script tries to infer it from the checkpoint dataset.",
    )
    parser.add_argument(
        "--out",
        help="Output bundle path. Defaults to `<ckpt stem>.ngviz` next to the checkpoint.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint or "model_args" not in checkpoint:
        raise ValueError(f"{path} does not look like a nanoGPT checkpoint.")
    return checkpoint


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def strip_unwanted_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unwanted_prefix = "_orig_mod."
    cleaned: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        cleaned_name = name[len(unwanted_prefix) :] if name.startswith(unwanted_prefix) else name
        cleaned[cleaned_name] = tensor
    return cleaned


def infer_meta_path(ckpt_path: Path, checkpoint: dict[str, Any], explicit_meta: str | None) -> Path | None:
    if explicit_meta:
        meta_path = Path(explicit_meta)
        if not meta_path.exists():
            raise FileNotFoundError(f"meta file not found: {meta_path}")
        return meta_path

    dataset = checkpoint.get("config", {}).get("dataset")
    if dataset:
        repo_root = ckpt_path.parent.parent
        candidate = repo_root / "data" / dataset / "meta.pkl"
        if candidate.exists():
            return candidate

    sibling = ckpt_path.with_name("meta.pkl")
    if sibling.exists():
        return sibling
    return None


def normalize_itos(raw_itos: Any) -> list[str]:
    if isinstance(raw_itos, list):
        return [str(token) for token in raw_itos]
    if isinstance(raw_itos, dict):
        return [str(token) for _, token in sorted(raw_itos.items(), key=lambda item: int(item[0]))]
    raise TypeError(f"Unsupported `itos` type: {type(raw_itos)!r}")


def load_tokenizer_manifest(meta_path: Path | None) -> dict[str, Any] | None:
    if meta_path is None:
        return None

    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)

    itos = normalize_itos(meta["itos"])
    stoi = meta.get("stoi", {})
    tokenizer_kind = "word" if meta.get("tokenizer") == "word" else "char"

    special_ids = {}
    if isinstance(stoi, dict):
        for token in ("<eos>", "<unk>", "<|endoftext|>"):
            if token in stoi:
                special_ids[token] = int(stoi[token])

    manifest = {
        "kind": tokenizer_kind,
        "itos": itos,
        "vocab_size": int(meta.get("vocab_size", len(itos))),
        "special_ids": special_ids,
        "meta_path": str(meta_path),
    }

    if tokenizer_kind == "word":
        manifest["token_regex"] = meta["token_regex"]

    return manifest


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().to(torch.float32).contiguous().cpu().numpy().tobytes(order="C")


def build_manifest(
    ckpt_path: Path,
    checkpoint: dict[str, Any],
    tokenizer_manifest: dict[str, Any] | None,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    model_args = {key: to_jsonable(value) for key, value in checkpoint["model_args"].items()}
    first_mlp = state_dict.get("transformer.h.0.mlp.c_fc.weight")
    mlp_hidden_size = int(first_mlp.shape[0]) if first_mlp is not None else None

    tensor_records = []
    blobs: list[tuple[str, bytes]] = []
    offset = 0
    parameter_count = 0

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        blob = tensor_to_bytes(tensor)
        numel = tensor.numel()
        parameter_count += numel
        tensor_records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": MANIFEST_DTYPE,
                "offset": offset,
                "nbytes": len(blob),
                "numel": numel,
            }
        )
        blobs.append((name, blob))
        offset += len(blob)

    n_embd = int(model_args["n_embd"])
    n_head = int(model_args["n_head"])
    manifest = {
        "format": "nanogpt-browser-bundle",
        "version": 1,
        "source": {
            "checkpoint_path": str(ckpt_path),
            "iter_num": int(checkpoint.get("iter_num", 0)),
            "best_val_loss": to_jsonable(checkpoint.get("best_val_loss")),
            "dataset": to_jsonable(checkpoint.get("config", {}).get("dataset")),
        },
        "model": {
            "block_size": int(model_args["block_size"]),
            "vocab_size": int(model_args["vocab_size"]),
            "n_layer": int(model_args["n_layer"]),
            "n_head": n_head,
            "n_embd": n_embd,
            "head_size": n_embd // n_head,
            "mlp_hidden_size": mlp_hidden_size,
            "dropout": float(model_args.get("dropout", 0.0)),
            "bias": bool(model_args.get("bias", True)),
            "parameter_count": int(parameter_count),
            "tensors": tensor_records,
        },
        "tokenizer": tokenizer_manifest,
    }
    return manifest, blobs


def write_bundle(out_path: Path, manifest: dict[str, Any], blobs: list[tuple[str, bytes]]) -> None:
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with out_path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", len(manifest_bytes)))
        handle.write(manifest_bytes)
        for _, blob in blobs:
            handle.write(blob)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".ngviz")
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = strip_unwanted_prefix(checkpoint["model"])
    meta_path = infer_meta_path(ckpt_path, checkpoint, args.meta)
    tokenizer_manifest = load_tokenizer_manifest(meta_path)
    manifest, blobs = build_manifest(ckpt_path, checkpoint, tokenizer_manifest, state_dict)
    write_bundle(out_path, manifest, blobs)

    model = manifest["model"]
    print(f"wrote {out_path}")
    print(
        "model:"
        f" layers={model['n_layer']}"
        f" heads={model['n_head']}"
        f" embd={model['n_embd']}"
        f" vocab={model['vocab_size']}"
        f" params={model['parameter_count']:,}"
    )
    if tokenizer_manifest is None:
        print("tokenizer: not found, UI will fall back to token-id input mode")
    else:
        print(f"tokenizer: {tokenizer_manifest['kind']} ({tokenizer_manifest['vocab_size']} tokens)")


if __name__ == "__main__":
    main()
