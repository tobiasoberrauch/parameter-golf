# RunPod Launch-Befehle

## Setup (einmalig nach SSH-Login)

```bash
cd /workspace
git clone https://github.com/tobiasoberrauch/parameter-golf.git
cd parameter-golf
git checkout my-submission

# Daten herunterladen (SP8192, volle 80 Shards)
python3 data/cached_challenge_fineweb.py --variant sp8192
```

---

## Track A: Incremental SOTA

### Einzelner Test-Run (1×H100, ~20 Min)

```bash
cd /workspace/parameter-golf
SEED=42 \
RUN_ID=track_a_seed42 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py
```

### Leaderboard-Run (8×H100, 3 Seeds)

```bash
cd /workspace/parameter-golf

# Seed 42
SEED=42 RUN_ID=track_a_seed42 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py 2>&1 | tee train_seed42.log

# Seed 314
SEED=314 RUN_ID=track_a_seed314 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py 2>&1 | tee train_seed314.log

# Seed 999
SEED=999 RUN_ID=track_a_seed999 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py 2>&1 | tee train_seed999.log
```

### Ergebnis prüfen

```bash
grep "final_" train_seed*.log
# Erwartete Ausgabe: final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
# Wenn TTT aktiv: final_ttt val_loss:X.XXXX val_bpb:X.XXXX
```

### TTT deaktivieren (falls Eval zu langsam)

```bash
SEED=42 RUN_ID=track_a_no_ttt TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py
```

---

## Track B: Adaptive Recurrence

```bash
# 8×H100, 3 Seeds
for SEED in 42 314 999; do
  SEED=$SEED RUN_ID=track_b_seed${SEED} \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-17_TrackB_AdaptiveRecurrence/train_gpt.py 2>&1 \
    | tee track_b_seed${SEED}.log
done
```

---

## Track C: Mamba SSM (Non-Record)

```bash
# SP8192 Daten für Track C (falls gewünscht, sonst sp1024)
SEED=42 RUN_ID=track_c_mamba \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-17_TrackC_SubMamba50M/train_gpt.py 2>&1 \
  | tee track_c_mamba.log
```

---

## Hyperparameter-Tuning (optional)

### QK-Gain Sweep (Track A)

```bash
for QK in 5.0 5.25 5.5 5.75 6.0; do
  SEED=42 RUN_ID=qk_sweep_${QK} QK_GAIN_INIT=$QK \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py 2>&1 \
    | tee qk_sweep_${QK}.log
done
grep "final_" qk_sweep_*.log
```

### TTT LR Sweep

```bash
for LR in 0.003 0.005 0.007 0.01; do
  SEED=42 RUN_ID=ttt_lr_${LR} TTT_LR=$LR \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/train_gpt.py 2>&1 \
    | tee ttt_lr_${LR}.log
done
```

---

## Kosten-Schätzung

| GPU | Preis | Pro Run (~20 Min) | 3 Seeds | QK-Sweep (5 Runs) |
|-----|-------|--------------------|---------|--------------------|
| 1×H100 | ~$3/h | ~$1 | ~$3 | ~$5 |
| 8×H100 | ~$20/h | ~$7 | ~$21 | ~$35 |

### RunPod Template

Nutze das offizielle Template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th

Wähle **8× H100 SXM** für finale Leaderboard-Runs.

---

## Nach dem Run: Submission vorbereiten

```bash
# Logs in Submission-Ordner kopieren
cp train_seed*.log records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/

# submission.json mit Ergebnissen aktualisieren
# val_bpb, seed_results, bytes_total ausfüllen

# LZMA-komprimiertes Script erstellen
cd records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/
python3 compress.py train_gpt.py > train_gpt_submit.py
mv train_gpt_submit.py train_gpt.py  # komprimierte Version als Submission

# PR erstellen
git add records/track_10min_16mb/2026-04-17_TrackA_IncrementalSOTA/
git commit -m "submission: SP8192 + 3L Recurrence + Parallel Residuals + QK5.5 + Adaptive TTT"
git push origin my-submission
gh pr create --title "SP8192 + 3L Recurrence + Parallel Residuals + QK5.5 + Adaptive TTT" \
  --body "## Summary
- 3-layer depth recurrence (L3-5, activate@0.35)
- Parallel residuals (GPT-J, layers 7+)
- QK-gain 5.5 (beyond SOTA 5.25)
- Score-first TTT with per-parameter-group LR
- SOTA hyperparameters (WD=0.095, EMA=0.9965)

## Results
[3-seed results here]

## Attribution
Built on @clarkkev PR #1394 stack"
```
