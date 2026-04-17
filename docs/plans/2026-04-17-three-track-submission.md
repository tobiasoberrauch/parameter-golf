# Three-Track Parameter Golf Submission Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Submit three parallel Parameter Golf entries — one incremental SOTA attempt, one with novel innovation, one experimental SSM architecture.

**Architecture:** All tracks share SP8192 tokenizer + GPTQ SDClip quantization base. Track A combines all known improvements + new delta. Track B adds adaptive recurrence. Track C implements a sub-50M Mamba-style SSM.

**Tech Stack:** PyTorch (8xH100), MLX (Apple Silicon local dev), SentencePiece SP8192, GPTQ, Brotli/LZMA compression, Flash Attention 3

**Base Code:** `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py` (58KB, 1408 lines)

**Current SOTA:** 1.0810 BPB (bigbag, 2026-04-09). Must beat by ≥0.005 nats at p<0.01.

---

## Track A: Incremental SOTA (Target: < 1.0760 BPB)

### Task A1: Setup submission directory and copy base stack

**Files:**
- Create: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/`
- Copy: base `train_gpt_human.py` from April 5 submission as starting point

**Step 1: Create directory structure**

```bash
mkdir -p records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA
```

**Step 2: Copy base stack**

```bash
cp records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py \
   records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py
```

**Step 3: Create submission.json skeleton**

```json
{
  "author": "Tobias Oberrauch",
  "github_id": "tobiasoberrauch",
  "name": "SP8192 + 3L Recurrence + Parallel Residuals + QK5.5 + Adaptive TTT",
  "date": "2026-04-17",
  "track": "10min_16mb",
  "val_bpb": null,
  "seeds": [42, 314, 999],
  "hardware": "8xH100 80GB SXM",
  "attribution": {
    "sp8192_gptq_sdclip": "@clarkkev (PR #1394)",
    "depth_recurrence": "@dexhunter (PR #1331, #1437)",
    "parallel_residuals": "@Robby955 (PR #1412)",
    "legal_ttt_framework": "@abaybektursun (PR #549), @dexhunter (PR #1413)"
  }
}
```

**Step 4: Commit**

```bash
git add records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/
git commit -m "feat(track-a): scaffold incremental SOTA submission from PR #1394 base"
```

---

### Task A2: Implement 3-layer depth recurrence (layers 3-5)

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Context:** Depth recurrence reuses physical layers 3,4,5 three times during forward pass, creating 17 virtual layers from 11 physical. No extra parameters — just forward pass reordering. Activates at training fraction 0.35.

**Step 1: Add recurrence config to Hyperparameters**

Add these fields:
```python
loop_layers: list[int] = [3, 4, 5]       # which layers to loop
num_loops: int = 3                         # total passes through loop segment
loop_activation_frac: float = 0.35         # training fraction when loops activate
```

**Step 2: Modify GPT.__init__ to compute layer indices**

After block creation, compute the full traversal order:
```python
# Build layer index sequence: [0,1,2, 3,4,5,3,4,5,3,4,5, 6,7,8,9,10]
encoder_base = list(range(self.num_encoder_layers))
decoder_base = list(range(self.num_encoder_layers, num_layers))
loop_seg = self.loop_layers  # [3,4,5]

# Replace loop_layers region with repeated segment
pre_loop = [i for i in encoder_base if i < loop_seg[0]]
post_loop_enc = [i for i in encoder_base if i > loop_seg[-1]]
self.loop_indices = pre_loop + loop_seg * num_loops + post_loop_enc + decoder_base
self.base_indices = list(range(num_layers))  # no-loop fallback
```

**Step 3: Modify GPT.forward to use loop indices conditionally**

```python
def forward_logits(self, input_ids, use_loops=True):
    indices = self.loop_indices if use_loops else self.base_indices
    # ... use indices instead of range(num_layers) for block traversal
```

**Step 4: Add activation scheduling in training loop**

```python
frac = step / total_steps
use_loops = frac >= args.loop_activation_frac
```

**Step 5: Smoke-test locally with MLX equivalent**

```bash
RUN_ID=track_a_recurrence ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  python3 train_gpt_mlx.py
```

Expected: Training runs without error, loss decreases.

**Step 6: Commit**

```bash
git commit -m "feat(track-a): add 3-layer depth recurrence (L3-5, 3x loop, activate@0.35)"
```

---

### Task A3: Implement parallel residuals (layers 7+)

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Context:** GPT-J style — from layer 7 onward, attention and MLP both read from the same pre-residual input (norm(x)) instead of sequential.

**Step 1: Add parallel_residuals config**

```python
parallel_residual_start: int = 7  # layers >= this use parallel residuals
```

**Step 2: Modify Block.forward to support parallel mode**

```python
def forward(self, x, x0, parallel=False):
    mix = self.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    if parallel:
        # GPT-J style: both read from same normalized input
        normed = self.attn_norm(x_in)
        attn_out = self.attn_scale[None, None, :] * self.attn(normed)
        mlp_out = self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x_in))
        return x_in + attn_out + mlp_out
    else:
        # Standard sequential
        attn_out = self.attn(self.attn_norm(x_in))
        x_out = x_in + self.attn_scale[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x_out))
        return x_out
```

**Step 3: Pass parallel flag in GPT.forward based on layer index**

```python
parallel = (layer_idx >= self.parallel_residual_start)
x = self.blocks[layer_idx](x, x0, parallel=parallel)
```

**Step 4: Smoke-test**

```bash
RUN_ID=track_a_parallel ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  python3 train_gpt_mlx.py
```

**Step 5: Commit**

```bash
git commit -m "feat(track-a): add GPT-J parallel residuals for layers 7+"
```

---

### Task A4: Tune QK-Gain to 5.5 (beyond current 5.25)

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Context:** QK-gain has shown monotonic improvement from 1.5→4.0→5.0→5.25. We push to 5.5.

**Step 1: Update default**

```python
qk_gain_init: float = 5.5  # was 5.25 in SOTA, 4.0 in baseline
```

**Step 2: Commit**

```bash
git commit -m "feat(track-a): increase QK-gain to 5.5 (monotonic trend beyond 5.25)"
```

---

### Task A5: Implement score-first legal TTT

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Context:** Test-time training on validation data. Must comply with 4 legal conditions: causal eval, standard softmax, score-before-update, single-pass.

**Step 1: Add TTT config**

```python
ttt_enabled: bool = True
ttt_chunk_size: int = 32768
ttt_epochs: int = 3
ttt_lr: float = 0.005
ttt_grad_clip: float = 1.0
```

**Step 2: Implement TTT eval function**

```python
def eval_with_ttt(model, val_tokens, args):
    """Score-first TTT: score chunk, then SGD-adapt, repeat."""
    all_losses = []
    all_byte_counts = []

    for chunk_start in range(0, len(val_tokens), args.ttt_chunk_size):
        chunk = val_tokens[chunk_start:chunk_start + args.ttt_chunk_size + 1]

        # PHASE 1: Score all windows FIRST (no gradients)
        with torch.no_grad():
            chunk_losses, chunk_bytes = sliding_window_eval(model, chunk)
            all_losses.extend(chunk_losses)
            all_byte_counts.extend(chunk_bytes)

        # PHASE 2: SGD adaptation on scored tokens
        optimizer = torch.optim.SGD(model.parameters(), lr=args.ttt_lr, momentum=0.9)
        for epoch in range(args.ttt_epochs):
            for batch in chunk_batches(chunk, args.train_seq_len):
                loss = model.loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.ttt_grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            # Cosine decay
            lr = args.ttt_lr * 0.5 * (1 + math.cos(math.pi * epoch / args.ttt_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

    return compute_bpb(all_losses, all_byte_counts)
```

**Step 3: Wire TTT into eval pipeline (after quantized roundtrip)**

The TTT runs on the already-quantized-and-dequantized model, during the eval phase.

**Step 4: Commit**

```bash
git commit -m "feat(track-a): implement legal score-first TTT (SGD 3ep, 32K chunks)"
```

---

### Task A6: Novel delta — per-parameter-group TTT learning rates

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Context:** Current TTT uses uniform SGD LR. We differentiate: embeddings get lower LR (already well-fit), attention matrices get higher LR (most room to adapt).

**Step 1: Replace uniform optimizer with parameter-group optimizer**

```python
ttt_optimizer = torch.optim.SGD([
    {'params': embed_params, 'lr': args.ttt_lr * 0.1},     # embeddings: conservative
    {'params': matrix_params, 'lr': args.ttt_lr * 1.5},    # attention/MLP: aggressive
    {'params': scalar_params, 'lr': args.ttt_lr * 0.5},    # scales/gains: moderate
], momentum=0.9)
```

**Step 2: Commit**

```bash
git commit -m "feat(track-a): differentiated TTT LR per parameter group"
```

---

### Task A7: Hyperparameter tuning (WD, EMA, matrix LR)

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py`

**Step 1: Apply known-best hyperparameters from SOTA submission**

```python
weight_decay: float = 0.095       # was 0.085
matrix_lr: float = 0.022          # was 0.02
ema_decay: float = 0.9965         # was 0.997
warmdown_frac: float = 0.72       # linear LR decay in final 72%
```

**Step 2: Commit**

```bash
git commit -m "feat(track-a): apply SOTA hyperparameters (WD=0.095, EMA=0.9965)"
```

---

### Task A8: LZMA compression wrapper + final artifact

**Files:**
- Create: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/compress.py`

**Step 1: Implement LZMA+base85 compression script**

```python
"""Compress train_gpt.py into self-extracting LZMA wrapper."""
import lzma, base64, sys
source = open(sys.argv[1]).read()
compressed = lzma.compress(source.encode(), format=lzma.FORMAT_RAW,
                           filters=[{"id": lzma.FILTER_LZMA2}])
encoded = base64.b85encode(compressed).decode()
wrapper = f'import lzma as L,base64 as B;exec(L.decompress(B.b85decode("{encoded}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))'
print(wrapper)
```

**Step 2: Verify artifact fits under 16MB**

```bash
python compress.py train_gpt.py > train_gpt_compressed.py
wc -c train_gpt_compressed.py  # must be << 16MB (code portion ~16KB)
```

**Step 3: Commit**

```bash
git commit -m "feat(track-a): add LZMA compression wrapper"
```

---

### Task A9: Create MLX local development version

**Files:**
- Create: `records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt_mlx.py`

**Context:** Mirror the PyTorch changes to MLX for local Apple Silicon iteration. This does NOT need TTT or GPTQ — just the architecture changes (recurrence, parallel residuals, QK-gain).

**Step 1: Copy baseline train_gpt_mlx.py and apply architecture changes**

- 3-layer depth recurrence (same loop logic)
- Parallel residuals for layers 7+
- QK-gain 5.5

**Step 2: Smoke-test**

```bash
RUN_ID=track_a_full ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  python3 records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt_mlx.py
```

**Step 3: Commit**

```bash
git commit -m "feat(track-a): MLX version with all architecture changes for local dev"
```

---

## Track B: Adaptive Innovation (Target: < 1.0800 BPB)

### Task B1: Setup submission directory from Track A base

**Files:**
- Create: `records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/`

**Step 1: Copy Track A base (after A1-A4 complete)**

```bash
mkdir -p records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence
cp records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py \
   records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/train_gpt.py
```

**Step 2: Create submission.json**

```json
{
  "author": "Tobias Oberrauch",
  "github_id": "tobiasoberrauch",
  "name": "SP8192 + Adaptive Recurrence + Per-Group SDClip",
  "date": "2026-04-17",
  "track": "10min_16mb"
}
```

**Step 3: Commit**

```bash
git commit -m "feat(track-b): scaffold adaptive recurrence submission"
```

---

### Task B2: Implement per-group SDClip with Hessian-aware k-values

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/train_gpt.py`

**Context:** Current SDClip uses uniform k=12.85 for all int6 matrices. Hessian trace analysis shows 30x variance between layer groups. Use group-level traces to allocate k non-uniformly: sensitive layers get higher k (wider clip = lower entropy = smaller compressed size), insensitive layers get lower k (tighter clip = lower distortion).

**Step 1: Add Hessian trace collection during GPTQ calibration**

```python
def collect_group_traces(model, calibration_data):
    """Compute per-layer-group Hessian trace estimates."""
    traces = {}
    for name, module in model.named_modules():
        if hasattr(module, 'hessian'):
            group = name.split('.')[1]  # e.g., 'blocks.3.attn.c_q' -> '3'
            traces[group] = traces.get(group, 0) + float(module.hessian.trace())
    return traces
```

**Step 2: Compute adaptive k per group**

```python
def adaptive_k_values(traces, base_k=12.85, lambda_adapt=0.175):
    """Modulate k per group: k_i = base_k * (1 + lambda * (trace_i/mean_trace - 1))"""
    mean_trace = sum(traces.values()) / len(traces)
    return {
        group: base_k * (1 + lambda_adapt * (t / mean_trace - 1))
        for group, t in traces.items()
    }
```

**Step 3: Pass per-group k into GPTQ quantization**

**Step 4: Smoke-test quantization size (must stay under 16MB)**

**Step 5: Commit**

```bash
git commit -m "feat(track-b): per-group SDClip with Hessian-aware k allocation"
```

---

### Task B3: Implement mixture-of-depths (conditional computation)

**Files:**
- Modify: `records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/train_gpt.py`

**Context:** Not all tokens need all layers. Add a lightweight router that skips computation for "easy" tokens, saving FLOPs → more training steps in 10 minutes.

**Step 1: Add router module**

```python
class DepthRouter(nn.Module):
    """Lightweight binary router: skip or process per token."""
    def __init__(self, dim, capacity_factor=0.5):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=False)
        self.capacity_factor = capacity_factor  # fraction of tokens that get processed

    def forward(self, x):
        scores = self.gate(x).squeeze(-1)  # (B, T)
        k = int(x.shape[1] * self.capacity_factor)
        topk_indices = scores.topk(k, dim=-1).indices
        return topk_indices, scores
```

**Step 2: Integrate router into selected blocks (layers 4-8)**

```python
def forward(self, x, x0, router=None):
    if router is not None:
        indices, scores = router(x)
        # Only process top-k tokens through attention+MLP
        x_selected = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        out = self._full_forward(x_selected, x0)
        # Scatter back
        x = x.scatter(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]), out)
        return x
    return self._full_forward(x, x0)
```

**Step 3: Tune capacity_factor (start with 0.75, sweep 0.5-0.9)**

**Step 4: Smoke-test: verify throughput increase (tok/s should increase)**

**Step 5: Commit**

```bash
git commit -m "feat(track-b): mixture-of-depths with lightweight routing (capacity=0.75)"
```

---

### Task B4: MLX local version for Track B

**Step 1: Port router + adaptive SDClip to MLX**

**Step 2: Smoke-test locally**

**Step 3: Commit**

---

## Track C: Experimental SSM (Non-Record Submission)

### Task C1: Setup non-record submission directory

**Files:**
- Create: `records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/`

**Step 1: Create directory**

```bash
mkdir -p records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M
```

**Step 2: Create submission.json**

```json
{
  "author": "Tobias Oberrauch",
  "github_id": "tobiasoberrauch",
  "name": "Sub-50M Mamba-2 SSM (First sub-50M byte-level SSM attempt)",
  "date": "2026-04-17",
  "track": "non_record_16mb",
  "note": "Experimental: validating whether state-space models work at 16MB scale"
}
```

**Step 3: Commit**

```bash
git commit -m "feat(track-c): scaffold experimental Mamba SSM submission"
```

---

### Task C2: Implement minimal Mamba-2 block in PyTorch

**Files:**
- Create: `records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py`

**Context:** Mamba-2 uses selective state-space layers instead of attention. Key components: input-dependent selection mechanism, hardware-aware scan algorithm. We implement a minimal version targeting ~30M params.

**Step 1: Implement SSM core**

```python
class SelectiveSSM(nn.Module):
    """Simplified Mamba-2 selective state space layer."""
    def __init__(self, dim, state_dim=16, dt_rank=None):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        dt_rank = dt_rank or math.ceil(dim / 16)

        # Selection mechanism: input-dependent discretization
        self.x_proj = nn.Linear(dim, dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, dim, bias=True)

        # SSM parameters
        A = torch.arange(1, state_dim + 1).float().repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim))

        # Projections
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Compute selection parameters
        x_dbl = self.x_proj(x_ssm)
        dt, B, C = torch.split(x_dbl, [self.dt_proj.in_features, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Discretize
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (b, l, d, n)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (b, l, d, n)

        # Selective scan (sequential for correctness, optimize later)
        h = torch.zeros(b, d, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(l):
            h = dA[:, i] * h + dB[:, i] * x_ssm[:, i].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)

        y = y + self.D * x_ssm
        y = y * F.silu(z)
        return self.out_proj(y)
```

**Step 2: Build MambaBlock and MambaLM**

```python
class MambaBlock(nn.Module):
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.ssm = SelectiveSSM(dim, state_dim)

    def forward(self, x):
        return x + self.ssm(self.norm(x))

class MambaLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, state_dim=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, state_dim) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)
        # tied embeddings

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = x @ self.tok_emb.weight.T
        return logits
```

**Step 3: Target architecture ~30M params**

```python
# Config: vocab=8192, dim=384, layers=12, state_dim=16
# ~30M params → fits in 16MB with int8+zlib
```

**Step 4: Integrate with existing data pipeline and eval (reuse from baseline)**

**Step 5: Smoke-test**

```bash
RUN_ID=track_c_mamba ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 \
  python3 records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py
```

**Step 6: Commit**

```bash
git commit -m "feat(track-c): minimal Mamba-2 SSM implementation (~30M params)"
```

---

### Task C3: Port Mamba to MLX for local development

**Files:**
- Create: `records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt_mlx.py`

**Step 1: Translate SelectiveSSM, MambaBlock, MambaLM to MLX**

Key MLX differences:
- `nn.RMSNorm` → use `rms_norm()` helper
- `F.silu` → `mx.sigmoid(x) * x`
- `torch.exp` → `mx.exp`
- Sequential scan → same logic with `mx.array`

**Step 2: Integrate with existing MLX data pipeline (copy from baseline)**

**Step 3: Smoke-test**

```bash
RUN_ID=track_c_mlx ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  python3 records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt_mlx.py
```

**Step 4: Commit**

```bash
git commit -m "feat(track-c): MLX port of Mamba-2 SSM for Apple Silicon"
```

---

### Task C4: Write README with analysis

**Files:**
- Create: `records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/README.md`

Document:
- Motivation: first sub-50M byte-level SSM attempt in Parameter Golf
- Architecture: Mamba-2 with selective scan, ~30M params
- Results: BPB comparison vs transformer baseline at same param count
- Analysis: where SSM helps/hurts at this scale
- Future work: hybrid SSM-transformer, hardware-optimized scan

---

## Execution Order & Dependencies

```
Track A (critical path):
  A1 → A2 → A3 → A4 → A5 → A6 → A7 → A8 → A9

Track B (after A4):
  B1 → B2 → B3 → B4

Track C (independent):
  C1 → C2 → C3 → C4

Parallelism:
  - A1-A4 first (shared foundation)
  - Then A5-A9, B1-B4, C1-C4 all in parallel
```

## Validation Criteria

- **Track A:** val_bpb < 1.0760, artifact < 16MB, 3 seeds with p<0.01
- **Track B:** val_bpb < 1.0800, artifact < 16MB, novel technique demonstrated
- **Track C:** Runs successfully, produces valid BPB score, interesting analysis
