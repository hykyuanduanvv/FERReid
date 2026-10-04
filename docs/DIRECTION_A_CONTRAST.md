# Direction A: contrastive context loss

目标：逼 LLM 生成"带有本域特征、真正提升本域检索"的 prompt，而不是一个与上下文无关的通用 prompt。

## 损失

每个 batch 来自一个训练域 d（`--batch_domain_mode single`；`--pseudo_domains camera_pair` 时域 = 数据集 × 摄像头对）。

1. 用本域上下文生成 prompt P_own，得到 batch 中查询身份的特征 x_own（原有流程不变）。
2. 从缓存 `Model._ctx_bank` 中随机取另一训练域 d' 最近一次的上下文（冻结的问题特征 + 标签，不重新编码图像），用同一个生成器生成 P_cross（带梯度，不更新 EMA 中心），在同一批图像上得到 x_cross。
3. 每个锚点的分离度 gap = d(最难正样本) − d(最难负样本)（FP32、关闭 autocast，见 `ops.losses.per_anchor_gap`），越小越好。
4. `ctx_contrast = mean(relu(margin + gap_own − gap_cross))`，总损失加上 `ctx_contrast_weight × ctx_contrast`。

第一次遇到某个域时只写入缓存、不计对比项。`--ctx_contrast_weight 0`（默认）时不走任何新分支，与原代码逐位一致。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ctx_contrast_weight` | 0 | > 0 启用；需要 `--model_type vicp`、`--batch_domain_mode single`、≥ 2 个训练域 |
| `--ctx_contrast_margin` | 0.05 | 本域 prompt 至少要比跨域 prompt 好这么多 |
| `--ctx_contrast_detach_encoder` | False | 跨域那一路只给上下文分支梯度；编码器 / LoRA / `base_prompt` / LLM 在这一路当常数（本域那一路照常训练） |
| `--init_from DIR` | "" | 训练前加载 checkpoint（`pytorch_model.bin` / `model.safetensors`）；VPT 的 `prompt` 变为残差模式的 `base_prompt` |
| `--train_context_only` | False | 只训练上下文分支（`mm_projector`、`query_embeddings`、`prompt_mlp`、`delta_norm`、`ctx_gate`） |
| `--delta_init_std 0` | — | 现在允许 0：`prompt_mlp` 置零，模型从 `base_prompt`（例如热启动的 VPT prompt）精确出发 |

日志里新增 `ctx_contrast`、`ctx_gap_own`、`ctx_gap_cross`。

## 推荐用法（从 VPT 热启动，只训上下文分支）

```bash
python scripts/train_reid.py --model_type vicp --backbone dinov2_b14 \
  --prompt_mode residual --ctx_center ema --delta_init_std 0 \
  --init_from <vpt_run>/checkpoint-XXXX --train_context_only True \
  --ctx_contrast_weight 1.0 --ctx_contrast_margin 0.05 \
  --batch_domain_mode single --pseudo_domains camera_pair ...
```

注意：不加 `--train_context_only` 时，对比项的梯度会经过 x_cross 流入编码器 / LoRA（让编码器"在外域 prompt 下更差"）。这是一条作弊捷径：编码器学会"不熟悉的 prompt → 特征变差"，而目标域的 prompt 对它来说正是不熟悉的。要么用 `--train_context_only`，要么在编码器也训练时加 `--ctx_contrast_detach_encoder`。
