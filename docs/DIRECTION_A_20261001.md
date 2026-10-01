# Direction A development results and numerical audit (2026-10-01)

This branch records completed development experiments, not a formal Protocol-2 or Protocol-3 main table. Production model defaults have not been replaced with the experimental LayerNorm or FP32 variants.

## Reproducibility fixes

- Seed before constructing the model. Historical runs seeded inside Trainer after model construction; their saved seed alone does not reproduce initialization.
- Preserve deliberately empty LOSS_ARGS (triplet-only) instead of falling back to BNNeck/CE.
- Deduplicate validation by training step before scoring the last three evaluations.
- Keep generated local chosen.sh/final.sh out of Git; reviewed experiment snapshots are included under results/direction_a_20261001.

## Results

Training uses Market+MSMT train splits; CUHK03 validation uses 500 fixed identities. Scores below are last-three-distinct-step validation mAP, one training seed.

| Configuration | Steps | mAP |
|---|---:|---:|
| VPT, triplet+WPA | 3000 | 54.03 |
| VPT, BNNeck+CE | 3000 | 47.91 |
| Plain LoRA, BNNeck+CE | 3000 | 43.84 |
| Full fine-tuning, BNNeck+CE | 3000 | 55.18 |
| VPT, cosine+warmup | 12000 | 57.05 |
| Full fine-tuning, cosine+warmup | 12000 | 54.74 |

Individual sweep candidates were margin=0.3 (+2.47), lr=3e-4 with cosine/warmup (+1.26), and LoRA on 12 layers (+1.01). The combination is unvalidated; the LR comparison also changes schedule versus the base run.

| Model (3000 steps) | VIPeR | GRID | i-LIDS |
|---|---:|---:|---:|
| VPT reference | 83.54 | 64.80 | 90.70 |
| a0 original-form generator | 8.37 | 2.34 | 28.25 |
| a1 residual bundle | 81.78 | 61.42 | 88.79 |
| a2 residual+EMA | 83.00 | 61.96 | 89.19 |
| a3 episode | 17.18 | 6.61 | 42.17 |
| a4 residual+EMA+episode | 82.57 | 61.57 | 88.48 |
| a5 all components | 9.23 | 2.26 | 27.66 |
| a6 residual+EMA+camera-pair | 7.71 | 2.07 | 28.71 |

These are own-domain k=16 prompt scores, averaged over 3 splits and 5 draws. Full k=4/16, cross-domain, and geometry data are archived in context_summary.json and CSVs. No configuration met the intended success criterion. Collapsed baselines must be investigated before treating these results as evidence against the method generally.

## a0 collapse audit

Old VICP checkpoint evaluated in the current pipeline: 83.07/62.67/88.72 on VIPeR/GRID/i-LIDS. This checks evaluation compatibility, not a matched-training comparison (old source data included all images).

| Isolated intervention | Steps | CUHK03 mAP |
|---|---:|---:|
| Current control | 100 | 0.37 |
| FP32 Triplet only | 100 | 0.41 |
| FP32 Triplet continuation | 300 | 0.35 |
| ICL loss weight 0 | 100 | 0.30 |
| Learning rate 1e-5 | 100 | 0.38 |
| Small random projection initialization | 100 | 0.38 |
| Normalize LLM hidden states before projection | 100 | 13.96 |

At control step 20, 86.4% of off-diagonal distances rounded to zero under AMP; FP32 on the same features produced 0% zeros. At step 100, 99.6% rounded to zero and the Triplet feature gradient was all zero. FP32 preserves gradients but did not recover accuracy by step 300. Hidden-state LayerNorm prevented the early numerical trap in the 100-step probe. This supports an upstream generator-conditioning issue plus downstream distance precision failure; it does not establish a unique root cause or validate a 3000-step repair.

Parameter gradient norms can be AMP-scaled when clipping is disabled; compare zero/nonzero behavior, not magnitudes across runs. The residual bundle also changes normalization, gating and initialization; its gain over a collapsed a0 is not proof that residual prompting or domain adaptation is responsible.

## Run the isolated probes

From the repository root with configs/local.sh configured:

```bash
source configs/local.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" scripts/launch_tasks.py plans/a0_diagnostics.tasks --gpus 0,1,2,7
```

The launcher respects existing success statuses. Use an isolated checkout/output archive for a fresh repetition; do not overwrite completed runs. Six 100-step jobs precede a barrier; the FP32 continuation resumes checkpoint-100 and records global trace steps using A0_PROBE_STEP_OFFSET=100. The probe runner uses runtime monkey patches only, never modifies production model files.

Remaining work: stable baseline reproduction, longer/multiple-seed LayerNorm tests, validated hyperparameter combination, per-fold source-only selection and explicit SYSU duplicate handling before formal P3. Direction B has not been run.
