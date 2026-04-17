# Sub-50M Mamba-2 SSM -- First byte-level SSM in Parameter Golf

## Motivation

Every submission in the Parameter Golf leaderboard uses a transformer (GPT-style
attention).  This experiment asks: **can a state-space model compete at the 16 MB
artifact scale?**

Mamba-2 is the natural candidate -- it replaces multi-head attention with a
selective state-space mechanism that is input-dependent (like attention) but runs
in O(N) time and constant memory per step at inference.  At ~30M parameters the
compressed artifact fits comfortably under 16 MB.

This is an *experimental, non-record* submission.  The goal is not to top the
leaderboard but to establish the first SSM baseline so future work can iterate.

## Architecture

| Component      | Detail                                    |
|----------------|-------------------------------------------|
| Type           | Mamba-2 Selective SSM                     |
| Vocab size     | 8192 (SentencePiece BPE)                 |
| Model dim      | 576                                       |
| Layers         | 12 MambaBlocks                            |
| State dim      | 16                                        |
| Inner dim      | 1152 (2x expansion)                       |
| dt rank        | 36                                        |
| Parameters     | ~30M (tied embeddings)                    |
| Logit softcap  | 30.0                                      |
| Optimizer      | Muon (matrices) + Adam (embeddings/1D)    |

Each MambaBlock: RMSNorm -> SelectiveSSM -> residual connection.

The SelectiveSSM uses:
- Input-dependent discretization (dt) via linear projection
- Input-dependent B, C projections (selection mechanism)
- Learnable A_log (log-space diagonal state matrix) and D (skip connection)
- Sequential scan over the sequence

## How to run

```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 \
  records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py

# 8x H100 (competition setting)
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py
```

Environment variables (same as baseline):
- `DATA_PATH` -- dataset directory (default: `./data/datasets/fineweb10B_sp8192`)
- `TOKENIZER_PATH` -- SentencePiece model (default: `./data/tokenizers/fineweb_8192_bpe.model`)
- `MAX_WALLCLOCK_SECONDS` -- training time cap (default: 600)

## Expected results

| Metric       | Value     | Notes                          |
|--------------|-----------|--------------------------------|
| val_bpb      | TBD       | First run pending              |
| artifact     | < 16 MB   | int8 + zlib                    |
| train time   | < 10 min  | 8x H100                       |

## How to run locally (Apple Silicon, MLX)

An MLX port (`train_gpt_mlx.py`) is provided for local development on Apple
Silicon Macs.  It uses sp1024 vocab by default (smaller dataset, faster
iteration) and a simplified Adam optimizer.

```bash
# Local MLX smoke test (Apple Silicon)
RUN_ID=mamba_smoke ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
  python3 records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt_mlx.py
```

MLX config defaults (differ from the PyTorch version):

| Setting    | MLX default | PyTorch default |
|------------|-------------|-----------------|
| vocab_size | 1024        | 8192            |
| model_dim  | 384         | 576             |
| dt_rank    | 24          | 36              |
| optimizer  | Adam        | Muon + Adam     |
| DATA_PATH  | sp1024      | sp8192          |

The MLX script includes the same int8+zlib quantization and roundtrip
validation as the PyTorch version.  All environment variables from the PyTorch
script are supported (ITERATIONS, MAX_WALLCLOCK_SECONDS, GRAD_ACCUM_STEPS,
etc.).

## Data pipeline

Uses the same binary-shard data loading, SentencePiece BPB evaluation, int8
quantization, and zlib compression as the baseline `train_gpt.py`.  Only the
model architecture is changed.
