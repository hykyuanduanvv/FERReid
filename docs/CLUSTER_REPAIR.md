# 簇修复：伪标签 + 主动合并/拆分查询

## 1. 动机

`Active_Pair_Prompt` 的 pair 查询里，每个"是/否"只影响 2 张图，各问题价值几乎一样，候选又是跨摄像头 kNN（本身就是很强的筛选），所以任何排序都很难稳定超过 `random`。本分支把人的回答从"标 2 张图"改为"修整个伪簇"：

- 每轮先对整个无标签候选池聚类得到伪身份（k-reciprocal Jaccard + DBSCAN，即 cluster-then-train 的 UDA 做法），域 token 在**全部**伪标签图像上训练；
- 人的回答作为约束作用于聚类：一个"是"合并两个伪簇（修复同一人按摄像头被拆开），一个"否"拆开一个不纯的簇；
- 一次回答可能影响几十张图，影响大小差异很大 —— 选择策略因此有了可利用的结构。

同时补上了零标注的无监督基线（`none`）。

## 2. 每一轮

| 步骤 | 内容 | 代码 |
|---|---|---|
| 1 | 当前 prompt 提取候选池特征 | `image_store.features` |
| 2 | k-NN（k1=30）、k-reciprocal 编码（局部扩展 k2=6）、稀疏 Jaccard 距离 | `pseudo.PoolGraph` |
| 3 | 按策略提问（见下），预算内；传递性可推出的跳过 | `loop._ask_pairs`、`repair.py` |
| 4 | DBSCAN（eps=0.6, min_samples=4）+ 约束：已回答的正簇涉及的伪簇整体合并；含 cannot-link 的伪簇按组拆分，其余成员归最近的组 | `PoolGraph.cluster` |
| 5 | 在全部伪簇上训练域 token：triplet + 簇中心记忆对比损失（温度 0.05，动量 0.2）；每簇两张图尽量来自两个摄像头；人工确认的 cannot-link 作困难负样本；token 跨轮继续（`--warm_start True`） | `prompt_tuning.py` |
| 6 | query/gallery 评测 | 不变 |

## 3. 问题与策略

两类问题，各是一个图像对：

- **合并**：两个伪簇的 medoid。候选为每簇最近的 `repair_k`(5) 个**摄像头不相交**的簇，加上 2 个任意摄像头的最近簇；离群图与其最近簇的 medoid 也算合并问题（影响 1 张图）。
- **拆分**：簇的 medoid 与其"远端部分"（比起 medoid 更接近簇内最远成员的那些成员）的 medoid。

`p` = 校准后的同人概率：一维 logistic，MAP 拟合已回答对，先验为以无标签阈值 tau 为中心的曲线（中心标准差 0.1，斜率对数标准差 1）。从第一个回答起就能更新，即使回答全是"是"。

| 策略 | 排序 | 角色 |
|---|---|---|
| `repair` | 期望改变的图像数：合并 `p·min(|A|,|B|)`，拆分 `(1−p)·|远端|` | **本方法** |
| `repair_unc` | `p(1−p)` | 消融：去掉影响项 |
| `repair_random` | 随机 | 同一问题池的随机下界 |
| `random` / `cover` / `disagree` | 原 kNN pair 问题，回答同样进入聚类约束 | 对照 |
| `none` | 不提问 | 无监督基线（0 次回答） |

每簇每轮最多参与 `repair_per_cluster`(1) 个问题。`disagree`（新）：kNN 对中余弦与 Jaccard 两种相似度排名分歧最大的先问（query by committee，偏向困难正样本）。

## 4. 合成数据上的预期（`tests/test_repair.py`，非真实结果）

60 人 × 3 摄像头、每人在每个摄像头下自成一簇的玩具域上，3 轮后伪标签的 pairwise F1（无训练）：

| 摄像头偏移 / 每轮预算 | repair | repair_unc | repair_random | random (kNN) | cover |
|---|---:|---:|---:|---:|---:|
| 0.6 / 15 | **0.924** | 0.865 | 0.731 | 0.815 | 0.819 |
| 0.6 / 40 | **1.000** | 0.967 | 0.821 | 0.924 | 0.907 |
| 0.9 / 15 | **0.734** | 0.644 | 0.578 | 0.670 | 0.659 |
| 0.9 / 40 | **0.865** | 0.827 | 0.653 | 0.814 | 0.827 |

真实数据上的差距要由 `plans/sim.tasks`（离线，分钟级）先确认，再跑训练实验。

## 5. 运行

```bash
python scripts/launch_tasks.py plans/sim.tasks --gpus 0,1,2            # 离线筛选（需 base_md_*）
python scripts/launch_tasks.py plans/repair_tune.tasks --gpus 0,1,2,3  # CUHK-SYSU 上选 eps / steps（需 base_tune）
python scripts/launch_tasks.py plans/repair.tasks --gpus 0,1,2,3,4,5,6,7
```

单独运行：

```bash
python scripts/eval_active.py --output_dir experiments/rep_try --checkpoint experiments/base_md_cuhk03/checkpoint-12000 \
    --domains cuhk03 --pseudo True --warm_start True --strategies none,repair,repair_random \
    --rounds 5 --budget 200 --steps 400 --n_seeds 1 --paired_ref repair_random --fp16 True --report_to none
```

## 6. 输出与判读

- `active.csv` 新列：`pw_prec/pw_rec/pw_f`、`nmi`、`pseudo_clusters`、`pseudo_outliers`、`pseudo_split_ids`（伪标签相对真实身份，只用于报告）；`n_merge_q/n_split_q/exp_change`（repair 本轮问了什么）；`ans_*`（只看人工回答形成的簇）。`true_ids/purity` 在 pseudo 模式下描述伪簇。
- `paired.csv`：同 seed 下相对 `--paired_ref` 的 mAP 差（均值、标准差、胜出 seed 数），以及 mAP-回答数曲线下面积（`round = aulc`，起点为 base）。
- `scripts/merge_active.py`：合并按 seed 拆开的任务。

判读顺序：

1. `sim_*`：`repair` 的 `pw_f` 是否明显高于 `repair_random` 与 `random`。不高就先别跑训练实验。
2. `none` vs base：无监督本身提升多少。这是审稿人最先问的基线。
3. `repair` vs `none`：1,000 次回答在无监督之上加了多少。
4. `repair` vs `repair_random` / `random`（`paired.csv`）：选择策略本身的贡献。
5. 消融：`rabl_nocontrast`、`rabl_nocross`、`rabl_nowarm`、`rabl_quota2`、`rabl_nomd`、预算曲线 `rabl_budget`。

## 7. 已知限制

- CUHK-SYSU 没有摄像头标签，调参折上跨摄像头采样与摄像头不相交合并不生效。
- 拆分问题只拆一个"远端部分"；更细的拆分需要多轮。
- 流式数据集（Market、MSMT17）训练时从全池采样，主机内存最多缓存 12,000 张（FP16，约 2.3 GB），其余每批现解码。
