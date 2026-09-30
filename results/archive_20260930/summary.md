## Oracle prompt, ViT-B/16 lr=1e-3

mean over splits (n=3)

| domain | opt_all | opt_k16 | opt_k4 | vicp | zero |
|---|---|---|---|---|---|
| viper | 79.04 | 71.64 | 70.34 | 68.97 | 46.13 |
| grid | 61.44 | 55.14 | 52.06 | 50.08 | 28.37 |
| ilids | 88.08 | 84.34 | 81.91 | 81.72 | 63.59 |

cross-domain (prompt tuned on train_domain, evaluated on domain), mAP

| domain | train_domain | rank1 | mAP |
|---|---|---|---|
| grid | ilids | 42.67 | 52.44 |
| grid | viper | 42.13 | 51.74 |
| ilids | grid | 75.56 | 82.50 |
| ilids | viper | 72.78 | 81.45 |
| viper | grid | 53.06 | 62.89 |
| viper | ilids | 59.60 | 69.07 |

prompt tuning: triplet loss start/end, relative prompt change

| condition | loss_start | loss_end | delta_norm |
|---|---|---|---|
| opt_all | 0.0793 | 0.0421 | 0.1199 |
| opt_k16 | 0.0547 | 0.0037 | 0.1150 |
| opt_k4 | 0.0207 | 0.0013 | 0.0634 |

h (LLM hidden state before prompt_mlp) vs prompt: {  "h": {   "within": 0.9784315360916985,   "across": 0.9627670384943485  },  "prompt": {   "within": 0.9970848394764794,   "across": 0.9951400694747766  },  "h_rel_std": 0.18641211092472076,  "prompt_rel_std": 0.06523551791906357 }
## Oracle prompt, ViT-B/16 lr=1e-2

mean over splits (n=3)

| domain | opt_all | opt_k16 | opt_k4 | vicp | zero |
|---|---|---|---|---|---|
| viper | 84.59 | 68.67 | 57.62 | 68.97 | 46.13 |
| grid | 64.49 | 52.40 | 48.10 | 50.08 | 28.37 |
| ilids | 84.34 | 79.18 | 78.99 | 81.72 | 63.59 |

cross-domain (prompt tuned on train_domain, evaluated on domain), mAP

| domain | train_domain | rank1 | mAP |
|---|---|---|---|
| grid | ilids | 34.40 | 44.00 |
| grid | viper | 36.53 | 46.40 |
| ilids | grid | 62.22 | 74.02 |
| ilids | viper | 67.22 | 77.52 |
| viper | grid | 39.77 | 50.01 |
| viper | ilids | 50.42 | 60.50 |

prompt tuning: triplet loss start/end, relative prompt change

| condition | loss_start | loss_end | delta_norm |
|---|---|---|---|
| opt_all | 0.0669 | 0.0098 | 0.3929 |
| opt_k16 | 0.0311 | 0.0005 | 0.2828 |
| opt_k4 | 0.0065 | 0.0005 | 0.2232 |

h (LLM hidden state before prompt_mlp) vs prompt: {  "h": {   "within": 0.9784315360916985,   "across": 0.9627670384943485  },  "prompt": {   "within": 0.9970848394764794,   "across": 0.9951400694747766  },  "h_rel_std": 0.18641211092472076,  "prompt_rel_std": 0.06523551791906357 }
## Oracle prompt, DINOv2-B/14 lr=1e-3

mean over splits (n=3)

| domain | opt_all | opt_k16 | opt_k4 | vicp | zero |
|---|---|---|---|---|---|
| viper | 89.37 | 83.36 | 83.60 | 82.99 | 60.63 |
| grid | 73.83 | 67.13 | 63.82 | 62.70 | 27.99 |
| ilids | 92.51 | 88.41 | 89.81 | 88.26 | 66.69 |

cross-domain (prompt tuned on train_domain, evaluated on domain), mAP

| domain | train_domain | rank1 | mAP |
|---|---|---|---|
| grid | ilids | 57.60 | 66.15 |
| grid | viper | 55.73 | 64.35 |
| ilids | grid | 82.78 | 88.29 |
| ilids | viper | 83.33 | 88.31 |
| viper | grid | 73.31 | 80.70 |
| viper | ilids | 75.53 | 82.53 |

prompt tuning: triplet loss start/end, relative prompt change

| condition | loss_start | loss_end | delta_norm |
|---|---|---|---|
| opt_all | 0.0576 | 0.0303 | 0.0977 |
| opt_k16 | 0.0418 | 0.0038 | 0.0953 |
| opt_k4 | 0.0164 | 0.0017 | 0.0533 |

h (LLM hidden state before prompt_mlp) vs prompt: {  "h": {   "within": 0.9636290967464447,   "across": 0.9367268097897371  },  "prompt": {   "within": 0.9977102246549394,   "across": 0.9956696157654127  },  "h_rel_std": 0.21555659174919128,  "prompt_rel_std": 0.06146201863884926 }

## Label value (frozen plain ViT features + KISSME)

| domain | condition | gamma | mAP |
|---|---|---|---|
| grid | center | -1.00 | 35.94 |
| grid | kiss_all | 0.10 | 34.43 |
| grid | kiss_all | 1.00 | 36.19 |
| grid | raw | -1.00 | 35.94 |
| grid | unlab | 0.10 | 35.02 |
| grid | unlab | 1.00 | 34.52 |
| ilids | center | -1.00 | 69.69 |
| ilids | kiss_all | 0.10 | 62.96 |
| ilids | kiss_all | 1.00 | 66.40 |
| ilids | raw | -1.00 | 69.69 |
| ilids | unlab | 0.10 | 68.19 |
| ilids | unlab | 1.00 | 67.74 |
| viper | center | -1.00 | 51.85 |
| viper | kiss_all | 0.10 | 54.71 |
| viper | kiss_all | 1.00 | 63.71 |
| viper | raw | -1.00 | 51.85 |
| viper | unlab | 0.10 | 51.95 |
| viper | unlab | 1.00 | 51.42 |

kiss_k: per split over 30 draws, then mean over splits (mAP)

| domain | gamma | k | mean | std | min | max | oracle-mean | mean-worst |
|---|---|---|---|---|---|---|---|---|
| grid | 0.10 | 2 | 35.12 | 0.81 | 33.45 | 36.82 | 1.70 | 1.67 |
| grid | 0.10 | 4 | 35.31 | 0.98 | 33.34 | 37.54 | 2.23 | 1.97 |
| grid | 0.10 | 8 | 35.60 | 1.19 | 33.03 | 38.16 | 2.56 | 2.57 |
| grid | 0.10 | 16 | 35.93 | 1.41 | 33.23 | 38.76 | 2.83 | 2.70 |
| grid | 0.10 | 32 | 36.12 | 1.39 | 33.26 | 39.00 | 2.88 | 2.86 |
| grid | 1.00 | 2 | 34.53 | 0.80 | 32.87 | 36.18 | 1.65 | 1.66 |
| grid | 1.00 | 4 | 34.67 | 0.95 | 32.72 | 36.73 | 2.07 | 1.95 |
| grid | 1.00 | 8 | 35.02 | 1.25 | 32.62 | 37.95 | 2.94 | 2.40 |
| grid | 1.00 | 16 | 35.48 | 1.40 | 32.72 | 38.28 | 2.81 | 2.75 |
| grid | 1.00 | 32 | 35.92 | 1.33 | 33.16 | 38.41 | 2.49 | 2.76 |
| ilids | 0.10 | 2 | 68.30 | 1.07 | 66.50 | 70.66 | 2.36 | 1.80 |
| ilids | 0.10 | 4 | 68.22 | 1.09 | 66.23 | 70.58 | 2.36 | 2.00 |
| ilids | 0.10 | 8 | 68.42 | 1.60 | 65.07 | 71.81 | 3.39 | 3.35 |
| ilids | 0.10 | 16 | 68.46 | 1.70 | 65.04 | 72.00 | 3.54 | 3.42 |
| ilids | 0.10 | 32 | 68.66 | 1.89 | 65.08 | 72.66 | 4.00 | 3.58 |
| ilids | 1.00 | 2 | 67.80 | 1.07 | 65.95 | 70.27 | 2.48 | 1.84 |
| ilids | 1.00 | 4 | 67.79 | 1.16 | 65.58 | 70.36 | 2.57 | 2.21 |
| ilids | 1.00 | 8 | 67.98 | 1.63 | 64.89 | 71.66 | 3.68 | 3.09 |
| ilids | 1.00 | 16 | 68.08 | 1.61 | 64.74 | 71.56 | 3.48 | 3.34 |
| ilids | 1.00 | 32 | 68.39 | 1.88 | 64.57 | 72.48 | 4.09 | 3.82 |
| viper | 0.10 | 2 | 52.45 | 0.58 | 51.23 | 53.57 | 1.12 | 1.22 |
| viper | 0.10 | 4 | 52.82 | 0.74 | 51.45 | 54.51 | 1.68 | 1.38 |
| viper | 0.10 | 8 | 53.55 | 0.84 | 51.82 | 55.46 | 1.91 | 1.73 |
| viper | 0.10 | 16 | 54.68 | 1.03 | 52.41 | 56.62 | 1.95 | 2.27 |
| viper | 0.10 | 32 | 55.65 | 1.04 | 53.58 | 57.96 | 2.31 | 2.08 |
| viper | 1.00 | 2 | 52.01 | 0.57 | 50.95 | 53.20 | 1.19 | 1.07 |
| viper | 1.00 | 4 | 52.38 | 0.68 | 51.15 | 53.88 | 1.50 | 1.23 |
| viper | 1.00 | 8 | 53.20 | 0.83 | 51.48 | 54.97 | 1.77 | 1.72 |
| viper | 1.00 | 16 | 54.57 | 1.05 | 52.34 | 56.64 | 2.07 | 2.23 |
| viper | 1.00 | 32 | 56.04 | 1.13 | 53.57 | 58.56 | 2.52 | 2.47 |

## Training (fold1 Market+MSMT -> val CUHK03, 3000 steps)

| model | best_mAP | best_step | final_mAP | mean_last5 |
|---|---|---|---|---|
| VICP ViT-B/16 | 34.39 | 2900 | 33.33 | 33.29 |
| VPT (fixed prompt) ViT-B/16 | 36.18 | 3000 | 36.18 | 35.65 |
| plain ViT-B/16 (LoRA only) | 25.23 | 2600 | 25.02 | 24.82 |
| VICP DINOv2-B/14 | 53.15 | 2100 | 52.77 | 52.66 |
| VPT DINOv2-B/14 | 54.44 | 2900 | 54.25 | 54.18 |
| plain DINOv2-B/14 | 46.82 | 3000 | 46.82 | 45.84 |
| VICP DINOv2-B/14 seed1 | 53.21 | 3000 | 53.21 | 52.69 |
| VPT DINOv2-B/14 seed1 | 55.06 | 3000 | 55.06 | 54.33 |

VICP ViT-B/16: last 200 steps id_loss 0.0361 icl_loss 0.4489 ot_loss -0.1679 std 0.0284

VPT (fixed prompt) ViT-B/16: last 200 steps id_loss 0.0365 icl_loss 0.0000 ot_loss -0.2366 std 0.0283

plain ViT-B/16 (LoRA only): last 200 steps id_loss 0.0561 icl_loss 0.0000 ot_loss 0.0000 std 0.0252

VICP DINOv2-B/14: last 200 steps id_loss 0.0224 icl_loss 0.5218 ot_loss -0.2150 std 0.0299

VPT DINOv2-B/14: last 200 steps id_loss 0.0216 icl_loss 0.0000 ot_loss -0.2275 std 0.0302

plain DINOv2-B/14: last 200 steps id_loss 0.0297 icl_loss 0.0000 ot_loss 0.0000 std 0.0301

VICP DINOv2-B/14 seed1: last 200 steps id_loss 0.0240 icl_loss 0.5410 ot_loss -0.2154 std 0.0298

VPT DINOv2-B/14 seed1: last 200 steps id_loss 0.0221 icl_loss 0.0000 ot_loss -0.2299 std 0.0308

## Selection sensitivity: ViT-B/16, k=16, random selections (VICP in-context vs prompt tuned on the same pairs)

| domain | generator | mean | std | min | max | best-mean | mean-worst |
|---|---|---|---|---|---|---|---|
| viper | VICP in-context | 68.73 | 0.20 | 68.45 | 69.01 | 0.28 | 0.28 |
| grid | VICP in-context | 49.98 | 0.30 | 49.58 | 50.67 | 0.69 | 0.40 |
| ilids | VICP in-context | 80.00 | 0.20 | 79.62 | 80.28 | 0.28 | 0.38 |
| viper | prompt tuned on k pairs | 71.10 | 0.39 | 70.56 | 71.77 | 0.67 | 0.53 |
| grid | prompt tuned on k pairs | 54.93 | 2.09 | 53.08 | 59.66 | 4.73 | 1.85 |
| ilids | prompt tuned on k pairs | 81.66 | 4.22 | 76.57 | 89.13 | 7.47 | 5.09 |

(n_splits=1, draws per split=10)

correlation over draws (vicp vs tuned): grid -0.09, ilids 0.52, viper 0.70

## Selection sensitivity: ViT-B/16, k=16, FIXED selection (generator noise) (VICP in-context vs prompt tuned on the same pairs)

| domain | generator | mean | std | min | max | best-mean | mean-worst |
|---|---|---|---|---|---|---|---|
| viper | VICP in-context | 68.66 | 0.16 | 68.45 | 68.90 | 0.23 | 0.21 |
| grid | VICP in-context | 49.93 | 0.20 | 49.59 | 50.26 | 0.34 | 0.33 |
| ilids | VICP in-context | 79.86 | 0.21 | 79.56 | 80.12 | 0.27 | 0.30 |
| viper | prompt tuned on k pairs | 70.05 | 0.62 | 68.91 | 71.20 | 1.14 | 1.15 |
| grid | prompt tuned on k pairs | 54.07 | 0.79 | 53.30 | 55.46 | 1.39 | 0.76 |
| ilids | prompt tuned on k pairs | 82.99 | 0.92 | 81.41 | 84.28 | 1.28 | 1.58 |

(n_splits=1, draws per split=10)

correlation over draws (vicp vs tuned): grid 0.07, ilids -0.21, viper -0.06

## Selection sensitivity: DINOv2, k=16, random selections (VICP in-context vs prompt tuned on the same pairs)

| domain | generator | mean | std | min | max | best-mean | mean-worst |
|---|---|---|---|---|---|---|---|
| viper | VICP in-context | 82.99 | 0.13 | 82.81 | 83.20 | 0.22 | 0.17 |
| grid | VICP in-context | 62.59 | 0.19 | 62.36 | 62.88 | 0.29 | 0.23 |
| ilids | VICP in-context | 88.78 | 0.24 | 88.33 | 89.01 | 0.23 | 0.45 |
| viper | prompt tuned on k pairs | 84.81 | 1.14 | 82.66 | 86.62 | 1.82 | 2.15 |
| grid | prompt tuned on k pairs | 67.74 | 1.52 | 64.44 | 69.46 | 1.72 | 3.30 |
| ilids | prompt tuned on k pairs | 89.32 | 1.95 | 85.45 | 91.98 | 2.66 | 3.88 |

(n_splits=3, draws per split=10)

correlation over draws (vicp vs tuned): grid 0.70, ilids 0.54, viper 0.16

## Selection sensitivity: DINOv2, k=16, FIXED selection (generator noise) (VICP in-context vs prompt tuned on the same pairs)

| domain | generator | mean | std | min | max | best-mean | mean-worst |
|---|---|---|---|---|---|---|---|
| viper | VICP in-context | 83.98 | 0.08 | 83.82 | 84.14 | 0.16 | 0.17 |
| grid | VICP in-context | 60.81 | 0.24 | 60.55 | 61.15 | 0.33 | 0.26 |
| ilids | VICP in-context | 89.97 | 0.39 | 89.15 | 90.22 | 0.25 | 0.82 |
| viper | prompt tuned on k pairs | 83.22 | 0.39 | 82.70 | 83.73 | 0.51 | 0.52 |
| grid | prompt tuned on k pairs | 64.20 | 0.55 | 63.44 | 65.37 | 1.18 | 0.76 |
| ilids | prompt tuned on k pairs | 92.34 | 0.81 | 90.36 | 93.11 | 0.77 | 1.98 |

(n_splits=1, draws per split=10)

correlation over draws (vicp vs tuned): grid 0.67, ilids -0.22, viper 0.16

## Selection sensitivity: DINOv2, k=4, random selections (VICP in-context vs prompt tuned on the same pairs)

| domain | generator | mean | std | min | max | best-mean | mean-worst |
|---|---|---|---|---|---|---|---|
| viper | VICP in-context | 84.08 | 0.15 | 83.82 | 84.22 | 0.14 | 0.26 |
| grid | VICP in-context | 60.90 | 0.26 | 60.60 | 61.23 | 0.32 | 0.30 |
| ilids | VICP in-context | 89.86 | 0.35 | 89.11 | 90.22 | 0.36 | 0.75 |
| viper | prompt tuned on k pairs | 83.37 | 0.86 | 81.79 | 84.52 | 1.15 | 1.58 |
| grid | prompt tuned on k pairs | 63.27 | 1.06 | 61.94 | 65.30 | 2.03 | 1.33 |
| ilids | prompt tuned on k pairs | 89.51 | 1.08 | 87.60 | 91.07 | 1.56 | 1.91 |

(n_splits=1, draws per split=10)

correlation over draws (vicp vs tuned): grid 0.26, ilids 0.06, viper -0.45

### std of tuned mAP over draws: random selections vs fixed selection (noise)

| domain | dinov2_k16 | dinov2_k16_noise | dinov2_k4 | vit_k16 | vit_k16_noise |
|---|---|---|---|---|---|
| grid | 1.52 | 0.55 | 1.06 | 2.09 | 0.79 |
| ilids | 1.95 | 0.81 | 1.08 | 4.22 | 0.92 |
| viper | 1.14 | 0.39 | 0.86 | 0.39 | 0.62 |

## Few-shot prompt tuning lr, chosen on the validation domain CUHK03 (DINOv2, 5 draws)

| lr | k | vicp_mAP | tuned_mAP | gain | tuned_std |
|---|---|---|---|---|---|
| 3e-4 | 4 | 47.18 | 47.23 | 0.05 | 0.40 |
| 3e-4 | 16 | 47.13 | 48.27 | 1.15 | 0.68 |
| 1e-3 | 4 | 47.18 | 46.58 | -0.60 | 1.27 |
| 1e-3 | 16 | 47.13 | 47.86 | 0.73 | 1.18 |
| 3e-3 | 4 | 47.18 | 45.88 | -1.30 | 1.48 |
| 3e-3 | 16 | 47.13 | 47.28 | 0.15 | 1.15 |

## Target domains, 10 splits, random k=16 context, 3 seeds (3000-step models; fold1 unless FINAL)

| model | domain | rank1 | mAP | mAP_seed_std |
|---|---|---|---|---|
| plain ViT-B/16 | grid | 29.28 | 39.61 | 0.00 |
| plain ViT-B/16 | ilids | 60.17 | 70.88 | 0.00 |
| plain ViT-B/16 | viper | 46.30 | 56.31 | 0.00 |
| VPT ViT-B/16 | grid | 41.52 | 52.99 | 0.00 |
| VPT ViT-B/16 | ilids | 72.67 | 80.57 | 0.00 |
| VPT ViT-B/16 | viper | 57.63 | 67.34 | 0.00 |
| VICP ViT-B/16 | grid | 41.33 | 51.10 | 0.10 |
| VICP ViT-B/16 | ilids | 74.22 | 82.08 | 0.18 |
| VICP ViT-B/16 | viper | 59.28 | 68.76 | 0.04 |
| plain DINOv2 | grid | 39.76 | 50.99 | 0.00 |
| plain DINOv2 | ilids | 76.83 | 83.98 | 0.00 |
| plain DINOv2 | viper | 71.65 | 78.88 | 0.00 |
| VPT DINOv2 | grid | 58.56 | 66.41 | 0.00 |
| VPT DINOv2 | ilids | 82.50 | 87.74 | 0.00 |
| VPT DINOv2 | viper | 79.30 | 85.23 | 0.00 |
| VICP DINOv2 | grid | 56.19 | 64.28 | 0.05 |
| VICP DINOv2 | ilids | 81.50 | 87.31 | 0.07 |
| VICP DINOv2 | viper | 75.80 | 83.21 | 0.05 |
| VPT DINOv2 seed1 | grid | 53.36 | 62.56 | 0.00 |
| VPT DINOv2 seed1 | ilids | 83.83 | 88.81 | 0.00 |
| VPT DINOv2 seed1 | viper | 77.12 | 83.65 | 0.00 |
| VICP DINOv2 seed1 | grid | 54.05 | 62.70 | 0.06 |
| VICP DINOv2 seed1 | ilids | 81.28 | 86.90 | 0.06 |
| VICP DINOv2 seed1 | viper | 74.06 | 81.82 | 0.04 |
| FINAL VPT DINOv2 (3 sources) | grid | 58.24 | 66.84 | 0.00 |
| FINAL VPT DINOv2 (3 sources) | ilids | 83.17 | 88.86 | 0.00 |
| FINAL VPT DINOv2 (3 sources) | viper | 79.81 | 86.53 | 0.00 |
| FINAL VICP DINOv2 (3 sources) | grid | 54.29 | 63.23 | 0.20 |
| FINAL VICP DINOv2 (3 sources) | ilids | 82.94 | 88.53 | 0.06 |
| FINAL VICP DINOv2 (3 sources) | viper | 77.57 | 84.78 | 0.02 |
