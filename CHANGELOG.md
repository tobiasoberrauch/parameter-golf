# CHANGELOG

## 2026-04-17 — tiny-llm-quantization
- Initialized deep research plan for extreme quantization below int6 on 80M-100M small language models with 16MB target.
- Collected round-1 evidence from primary papers (AQLM, QuIP#, LLM-QAT, BitNet b1.58 Reloaded), official repos/docs, and local parameter-golf records for int6 GPTQ+SDClip and QAT baselines.
- Verified explicit 16MB nominal storage math: 80M/90M/100M params exceed budget at int6, int4, int3, and usually int2; 80M fits only around 1.58 bits nominal, 90M-100M do not.
- Wrote quantitative research notes and two figures: nominal storage feasibility vs 16 MiB and practical 16MB frontier from local artifact-constrained runs.
- Drafted and finalized the research brief at `outputs/tiny-llm-quantization.md` and provenance at `outputs/tiny-llm-quantization.provenance.md`.
- Reviewer verification pass succeeded and flagged citation-traceability / confidence-calibration issues; these were fixed manually.
- Verifier subagent failed due missing Anthropic API key, so inline citation anchoring and final URL verification were completed manually.

## 2026-04-17 — sp8192-stack-comparison
- Began a source comparison across three SP8192 record variants: 2026-04-05 base stack, 2026-04-08 parallel residual + score-first TTT, and 2026-04-09 legal TTT + 3-layer recurrence.
- Attempted a `researcher` subagent for evidence gathering, but the subagent runtime again lacked the needed API/tooling; fell back to manual evidence collection.
- Decoded all three compressed `train_gpt.py` wrappers and compared them against README and submission metadata.
- Verified key mismatches already: April 8 README claims QK-Gain 5.0 while code default stays at 4.0 and the repro command omits the override; April 9 README claims QK-Gain 5.25 while code default is 5.0 and only the repro command applies 5.25.
- Verified that April 8 and April 9 implement different notions of "parallel residuals" (lane-merge vs in-block GPT-J-style parallel attention/MLP), so they should not be treated as identical methods.
- Wrote intermediate evidence notes at `outputs/notes/sp8192-stack-comparison-evidence.md`, drafted the final comparison, and generated quantitative comparison figures under `outputs/figures/`.
- Attempted the required `verifier` subagent pass for inline citation/URL verification, but it failed because the subagent runtime lacks an Anthropic API key; finalized the comparison manually at `outputs/sp8192-stack-comparison-comparison.md`.
