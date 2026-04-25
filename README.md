# nanoGPT Signal Lab

## Usage

1. Convert a checkpoint:

```sh
python tools/convert_checkpoint.py --ckpt out-shakespeare-char/ckpt.pt
```

2. Serve the repo root:

```sh
python -m http.server 8000
```

3. Open `http://localhost:8000/inspector/`

4. Drop the generated `.ngviz` bundle into the UI, type a prompt, and click `Step Once`.

## Notes

- The browser runtime is shape-driven and does not instantiate `model.py`.
- Tokenizer metadata is embedded when `meta.pkl` is available.
- The runtime captures intermediate tensors for embeddings, attention, residual adds, MLPs, final logits, and sampling.
- This is a CPU implementation intended for small to medium nanoGPT checkpoints, not full GPT-2 class models.
