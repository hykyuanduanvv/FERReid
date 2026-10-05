# 主动 pair 查询 + 目标域 prompt

## 1. 设定

一个在源域训练好的 ReID 模型（基础模型，默认 DINOv2 VPT）部署到新的摄像头网络：

- 目标域候选池（train 划分）**没有身份标签**，**有摄像头标签**（监控数据自带）；
- 标注员可以回答"这两张图是不是同一个人"（pair 查询，是/否）；
- 基础模型参数**全部冻结**，每个目标域只学一组 prompt 参数，学完后冻结、插入，成为该域的常参数；
- 成本 = 人工回答的次数（另报告最终覆盖的身份数）。

## 2. 每一轮做什么

| 步骤 | 内容 | 代码 |
|---|---|---|
| 1 | 用当前 prompt 提取候选池全部图像的特征 | `adapters/active/image_store.py` |
| 2 | 模型提出它认为"可能是同一个人"的 pair：每张图在**其他摄像头**中的 k 个近邻（默认 k=10），记录相似度、是否互为近邻 | `adapters/active/candidates.py` |
| 3 | 选择策略给候选 pair 排序；按顺序提问，答案能由传递性推出的 pair 跳过不计成本，直到用完本轮预算 | `adapters/active/pair_selection.py`、`loop.py` |
| 4 | 正对用 union-find 合并成身份簇；负对记为簇之间的 cannot-link（困难负样本） | `adapters/active/constraints.py` |
| 5 | 冻结基础模型，用簇和困难负样本训练域 prompt（triplet，FP32） | `adapters/active/prompt_tuning.py` |
| 6 | 在 query/gallery 上评测（可只评部分轮次） | `image_store.TargetSplit.evaluate` |

**传递性**：A~B 且 B~C ⇒ A~C；A~B 且 B≁C ⇒ A≁C。日志中的 `n_inferred` 是被跳过的 pair 数。

**判定阈值 tau**：第一轮用无标签规则（随机图对相似度的 99 分位数，随机对几乎都是不同的人）；之后用已回答的 pair 在当前特征上拟合（最大化平衡准确率）。

## 3. 选择策略

| 名称 | 做法 | 角色 |
|---|---|---|
| `random` | 候选 pair 随机顺序 | 下界 |
| `confident` | 相似度最高的先问（模型最确信的"同一人"） | 对照：预期信息量低 |
| `uncertain` | 相似度最接近 tau 的先问 | 经典不确定性 |
| `balanced` | tau 上下交替取最近的 | LBAS 式正负平衡 |
| `cover` | **我们的方法**：不确定性 + 覆盖。预算按 `--expand_ratio` 分成两部分：连接已标注图像的 pair（扩充或连接已有身份簇）和全新图像之间的 pair（发现新身份）；同一轮不问"看起来是同一个人"的重复 pair；每对摄像头有配额 | 主方法 |
| `anchor:<选择器>` | ID 级标注协议：选择器挑锚点图，标注员在另一个摄像头里找到这个人（`<选择器>` 为 `adapters/context_selection.IMAGE_SELECTORS` 中的方法，如 `random`、`facility_camera`） | 与 in-context 设定相同的标注方式；成本单位是"找人"而非"是/否" |
| `oracle_all` | 候选池全部身份都标注 | 上界 |

## 4. 域 prompt

- `--prompt_mode append`（默认）：基础 prompt（每层 V 个 token）不动，每层追加 `--domain_tokens` 个新 token 并训练，得到 (1, L, V+m, D)。换域只换这 m 个 token。
- `--prompt_mode replace`：训练基础 prompt 的一个副本（即之前实验中的 "tuned prompt"）。

每一步取 `--ids_per_batch` 个至少两张图的身份簇，每簇两张；以 `--hn_prob` 的概率把该簇的一个 cannot-link 簇一起放进 batch（人工确认的困难负样本）。默认每轮从基础 prompt 重新训练 `--steps 300` 步（lr 3e-4）。

## 5. 运行

```bash
python scripts/launch_tasks.py plans/diag.tasks --gpus 0,1,2,3
python scripts/launch_tasks.py plans/active_small.tasks --gpus 0,1,2,3
python scripts/launch_tasks.py plans/active_large.tasks --gpus 0,1,2
```

单独运行：

```bash
python scripts/eval_active.py --output_dir experiments/act_try --checkpoint <VPT checkpoint> \
    --domains viper --eval_splits 1 --n_seeds 1 --strategies cover,random --rounds 2 --budget 25 \
    --fp16 True --report_to none
```

小目标域图像缓存在显存；超过 `--cache_max`（默认 6000 张）的集合从磁盘流式读取，只有被标注的图像会解码进内存。

## 6. 输出

`experiments/<任务名>/`：

- `active.csv`：每个 domain / split / strategy / seed / round 一行。主要列：`n_queries`（是/否回答数）、`n_anchors`（找人次数）、`n_pos` / `n_neg` / `n_inferred`、`n_clusters`（≥2 张图的簇）、`true_ids`（簇覆盖的真实身份数，仅用于报告）、`purity`（簇纯度）、`split_ids`（同一人被分成多个簇的数目）、`tau`、`round_pos_rate`（本轮回答为"是"的比例）、`mAP`、`rank1`、prompt 训练损失与耗时。
- `summary.csv`：按 domain / strategy / round 对 split × seed 取平均。
- `prompts/*.pt`：每次运行最终的域 prompt（`{"prompt": (1, L, V', D), ...}`），即冻结后插入的常参数。

诊断（`plans/diag.tasks`）：`diag_retrieval.json`（hit@k、互为近邻精度、候选 pair 正样本率、按相似度十分位的正样本率、AUC）和 `verify_pairs.csv`（供 `scripts/diag_vlm.py` 比较多模态大模型与 ReID 模型的同人判断能力）。

## 7. 判读

1. **诊断先行**：`hit@10` 太低（例如 < 0.3）说明第一轮提出的 pair 大多是负对，`cover` 应提高 `--candidate_k` 或降低 `--expand_ratio`；十分位正样本率若从 0 陡升到 1、中间没有过渡带，说明不确定区间很窄，`uncertain` 与 `balanced` 会接近。
2. **主表**：同一预算（`n_queries`）下各策略的 mAP，相对 `base`（第 0 轮）与 `random` 的提升，距 `oracle_all` 的差距；`true_ids` 说明同样的回答数覆盖了多少人。
3. **成本口径**：pair 策略与 `anchor:*` 的成本单位不同。报告时两种都列，并说明一次"找人"通常比一次"是/否"贵。
4. **噪声尺度**：以 `random` 在不同 seed 间的标准差为尺度，小于它的差别不下结论。

## 8. 已知限制与后续

- `anchor:*` 跨轮次可能再次选中已标注的人，形成两个未合并的簇（`split_ids` 记录），这在真实 ID 级标注中不会发生，会略低估 anchor 协议。
- 适配方式的对照（同样标注下全参微调、LoRA、只调 BN）尚未实现；需要时在 `prompt_tuning.py` 旁加一个"可训练参数集合"的选项。
- 多模态大模型目前只作诊断（`scripts/diag_vlm.py`，未在本仓库的测试环境运行过），还没有接入为人工前的预筛选。
- 目标域的无标签图像只用于提 pair 和阈值，没有参与 prompt 训练（例如伪标签）；可作为后续扩展。
