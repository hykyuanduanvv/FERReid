# FERReID：少样本上下文行人重识别

**Few-shot Exemplar Recruitment for In-Context Person Re-Identification**

基于 VICP 的行人 ReID 研究代码：在源数据集训练模型，在新数据集中提供少量已标注的跨摄像头图像对，生成视觉 prompt；测试时冻结参数，完成 query-to-gallery 检索。研究目标是分析上下文信息的作用，并探索如何选择更有价值的支持样例。

> 当前是研究中的实验仓库，不表示论文已被 CVPR 2027 接收。已实现 VICP 行人适配、plain / VPT 基线、ViT-B/16 与 DINOv2-B/14 两种主干、三折源域交叉验证、上下文诊断，以及 2026-10-01 新增的强基线选项（BNNeck + ID 损失、全参数微调、PK 采样等）、Protocol-2 与严格 label-free 的按图选择；正式选择器仍只有 `first` 与 `random`，新的招募算法尚未实现。
>
> **2026-10-01 更新**：代码以 2026-09-30 服务器版本为基础（DINOv2、VPT、分析脚本）。9/30 的发现——DINOv2 远好于 ViT-B/16、固定 prompt（VPT）≥ VICP、域专属 prompt 上限大——见 [9/30 报告](results/archive_20260930/REPORT_2026-09-30.md)；下一步运行计划见 [RUN_PLAN_20261001](docs/RUN_PLAN_20261001.md)。**选择单位默认改为 `image`**：复现 9/30 及更早结果须加 `--selection_unit identity`。

## 导航

- [**下一步运行计划（强基线 + Protocol-2 + 方向 A，多卡）**](docs/RUN_PLAN_20261001.md)
- [9/30 实验报告（DINOv2、VPT 对照、域专属 prompt 上限、选人敏感性）](results/archive_20260930/REPORT_2026-09-30.md)
- [环境安装、权重准备与完整部署](docs/DEPLOYMENT.md)
- [模型架构与实验协议](docs/EXPERIMENTS.md)
- [历史结果、统计口径与局限](results/README.md)
- [代码来源与第三方说明](THIRD_PARTY_NOTICES.md)
- [移植改动与复现注意事项](docs/REPRODUCIBILITY.md)

## 方法概览

```mermaid
flowchart LR
    A[支持身份的跨摄像头图像对] --> B[冻结 ViT 提取 CLS]
    B --> C[Q-Former：每对 32 个视觉 token]
    C --> D[图像对 token + yes/no 答案]
    D --> E[冻结 Qwen3-0.6B]
    E --> F[384 个查询的隐状态 + 线性映射]
    F --> G[12 层 × 32 个视觉 prompt]
    H[Query 与 Gallery 行人图像] --> I[ViT-B/16 + LoRA]
    G --> I
    I --> J[归一化 CLS 特征与余弦排序]
```

| 组件 | 实际配置 |
|---|---|
| 视觉主干 | `--backbone vit_b16`（默认，历史）：timm ViT-B/16，256×128；`dinov2_b14`：DINOv2 ViT-B/14，模型内缩放到 252×126（18×9 patches）。均 12 层、768 维 |
| 参数高效微调 | 最后 4 层 Q/K/V 的 LoRA，rank=128 |
| 图像对编码 | 2 层 Q-Former，每对输出 32 个 token |
| 语言模型 | Qwen3-0.6B，参数冻结，隐藏维度 1024 |
| 视觉提示 | 每层 32 个 token，同一支持集的 prompt 供全部 query/gallery 共享 |
| 损失 | hardest triplet（margin=0.1）+ ICL 答案预测 + 0.01×WPA 局部对齐 |
| 可训练部分 | LoRA、Q-Former、提示查询、prompt 线性映射，约 37.95M 参数 |
| plain 基线 | 同一主干 + LoRA，只使用 triplet，约 2.36M 参数 |
| VPT 基线 | 与 VICP 相同的 LoRA / prompt 位置 / 损失，prompt 为一个固定可学习参数，不用 LLM 与上下文，约 2.65M 参数 |
| 可选（2026-10-01） | BNNeck + ID 交叉熵、全参数微调、LoRA 层数/秩、PK 与跨摄像头采样、混合域 batch；默认全部关闭（见 [运行计划](docs/RUN_PLAN_20261001.md)） |

`k` 是支持身份数，`num_icl_samples=64` 是上下文问题数，二者独立。历史实现逐题随机选择“同身份对”或“随机图像对”，随机对也可能同人，因此 **64 题不保证严格 32 正 / 32 负**。这与另外的物体 ReID 比例控制实验不同。

## 数据集

下表为服务器读取器检查的单个划分统计，记为“身份数 / 图像数”。路径均相对 `FERREID_DATA_ROOT`。图片与预训练权重不放入 Git。

| 数据集 | 主折角色 | train / 支持候选池 | query | gallery | 数据目录与状态 |
|---|---|---:|---:|---:|---|
| Market1501 | 源训练域 | 751 / 12,936 | 750 / 3,368 | 751 / 15,913 | `market1501/Market-1501-v15.09.15`，已准备 |
| MSMT17 V1 | 源训练域 | 1,041 / 30,248 | 3,060 / 11,659 | 3,060 / 82,161 | `msmt17/MSMT17_V1`，已准备 |
| CUHK03-NP labeled | 主折验证域 | 767 / 7,368 | 700 / 1,400 | 700 / 5,328 | `cuhk03/CUHK03_pytorch/CUHK03_pytorch`，已准备 |
| VIPeR | 目标测试域 | 316 / 632 | 316 / 316 | 316 / 316 | `viper/VIPeR`，已准备 |
| GRID | 目标测试域 | 125 / 250 | 125 / 125 | 126 个读取器 PID 值 / 900 | `grid/underground_reid`，已准备 |
| i-LIDS | 目标测试域 | 59 / 222–246 | 60 / 60 | 60 / 60 | `ilids/i-LIDS_Pedestrian`，已准备 |
| PRID2011 | 计划目标域 | — | — | — | `prid2011/prid_2011`，**缺失，未报告结果** |
| CUHK02 | 未纳入当前配置 | — | — | — | `cuhk02/Dataset`，**缺失** |
| CUHK-SYSU（ReID 裁剪版） | Protocol-2 | 文献值 5,532 / 15,088（待核对） | 2,900 / 2,900 | 2,900 / 5,447 | `cuhksysu/cuhksysu4reid/{train,query,gallery}`，**缺失**；无摄像头标签 |

- Market/MSMT/CUHK03 由已有数据包整理；VIPeR、i-LIDS 的历史数据来自官方发布包的存档；GRID 由 torchreid 读取器准备。新使用者应按发布方条款自行取得数据，仓库不重新分发图像。
- GRID 的 126 是读取器的不同 PID 值数量，包含干扰项编码，不能解释成仅有 126 个真实人物。i-LIDS 指静态行人图像数据，**不是 iLIDS-VID**。
- 默认 `source_all_images=True`：仅对源训练域合并 train/query/gallery。主折实际训练使用 Market **1,501 身份 / 29,419 图**与 MSMT **4,101 身份 / 126,441 图**，不能与仅使用官方 train 的工作直接比较。
- 验证域和目标域不做 combineall：train 仅作为支持候选池，query/gallery 用于评估。至少出现在两个摄像头的身份可作为支持：Market 751、MSMT 1,022、CUHK03 766；目标域每折 VIPeR 316、GRID 125、i-LIDS 59。
- VIPeR 使用记录划分 0–9；文件中的 10–19 是摄像头互换版本。GRID/i-LIDS 使用 10 个记录划分。这些由读取器生成或整理的划分不能笼统称作唯一“官方 split”。`data_manifests/` 保存历史结果对应的精确划分。
- 当前主流程检查候选池与 query/gallery 的**路径不重叠**，尚未完成本行人数据全量字节哈希及视觉近重复审计；不能据此声称已排除一切泄漏。

测试使用目标域少量身份标注，因此本协议应表述为“目标域少量标注、参数冻结的上下文 ReID”，不宜直接等同于完全不接触目标域标注的严格域泛化。

## 快速开始

先按 [部署文档](docs/DEPLOYMENT.md) 安装依赖、准备图像和权重。推荐 Linux + NVIDIA CUDA，历史机器为 RTX 4090 D 24GB。

```bash
git clone https://github.com/hykyuanduanvv/CVPR-2027.git
cd CVPR-2027
cp configs/paths.example.sh configs/local.sh
# 编辑路径后加载
source configs/local.sh
python scripts/install_splits.py --apply
python scripts/check_datasets.py

# 主折：Market + MSMT 训练，CUHK03 验证；各 1,000 步
bash scripts/run_main.sh vicp main_vicp 1000 42
bash scripts/run_main.sh plain main_plain 1000 42

# 完整模型：10 splits × 3 支持种子，k=16
python scripts/eval_context.py \
  --output_dir experiments/main_vicp/eval \
  --checkpoint experiments/main_vicp/val_cuhk03/checkpoint-1000 \
  --model_type vicp --domains viper,grid,ilids \
  --methods random --ks 16 --eval_seeds 3 --eval_splits 10 \
  --num_icl_samples 64 --fp16 True --report_to none --selection_unit identity
```

参数定义见 `adapters/args_reid.py`，完整命令、诊断条件和 3,000 步扩展见部署文档。

## 已有结果

以下来自服务器原始 CSV，属于**历史结果**，不是本次整理代码后重新训练的数值。主折均训练 1,000 步；VICP 使用 k=16、10 个记录划分、3 个支持种子；plain 使用同样 10 个划分、1 次确定性评估。

| 目标域 | Plain mAP | Plain Rank-1 | VICP mAP | VICP Rank-1 |
|---|---:|---:|---:|---:|
| VIPeR | 51.83 | 41.17 | 59.21 | 49.62 |
| GRID | 35.89 | 26.88 | 43.99 | 35.17 |
| i-LIDS | 69.68 | 58.83 | 76.07 | 67.17 |

单位为百分数。逐次结果、训练曲线和权重 SHA256 见 [结果目录](results/README.md)。plain 同时去掉上下文分支、WPA 和额外参数，不能将差值单独归因于支持集具体内容。当前没有多训练种子统计，也没有完成 3,000 步模型的同规格目标域主表。

**2026-09-30 结果**（fold1：Market+MSMT combineall 训练，3,000 步；目标域 10 个划分，mAP；VICP 为 k=16、3 个支持种子、按身份选择；详见 [9/30 报告](results/archive_20260930/REPORT_2026-09-30.md)）：

| 模型 | VIPeR | GRID | i-LIDS |
|---|---:|---:|---:|
| plain ViT-B/16 | 56.31 | 39.61 | 70.88 |
| VPT ViT-B/16 | 67.34 | 52.99 | 80.57 |
| VICP ViT-B/16 | 68.76 | 51.10 | 82.08 |
| plain DINOv2 | 78.88 | 50.99 | 83.98 |
| VPT DINOv2 | 85.23 | 66.41 | 87.74 |
| VICP DINOv2 | 83.21 | 64.28 | 87.31 |
| VPT DINOv2（Market+MSMT+CUHK03） | 86.53 | 66.84 | 88.86 |
| VICP DINOv2（Market+MSMT+CUHK03） | 84.78 | 63.23 | 88.53 |

单训练种子（DINOv2 另有 seed 1 复现，见报告）；目标域已被多项探索使用，不宜作为最终无偏评估。

```bash
# 只读取随仓库附带的记录；无需 GPU 或第三方 Python 依赖
python scripts/summarize_results.py
python scripts/verify_repository.py
```

## 目录

```text
adapters/           行人模型、数据读取、支持集选择和训练/评估适配
models.py           VICP 式 Q-Former → Qwen → 分层视觉 prompt
custom_trainer.py   多损失记录的 Trainer；主流程使用 DGReIDTrainer 子类
ops/                LoRA、triplet、WPA
scripts/            训练、评估、诊断、部署辅助、多卡启动器（launch_tasks.py）和核验入口
plans/              分阶段实验任务清单（*.tasks）与共享设置（common.sh）
tests/              CPU 自检：选择逻辑（与旧实现逐项比对）、各模型变体前后向
configs/            本地路径配置示例
docs/               中文部署、实验协议和复现记录
data_manifests/     历史 VIPeR/GRID/i-LIDS 划分；无图像
results/            历史 CSV、训练状态和来源指纹
env/                原服务器环境快照（仅用于追溯）
```

## 引用与代码来源

本项目改编自 [VICP: Generalizable Object Re-Identification via Visual In-Context Prompting](https://arxiv.org/abs/2508.21222)（ICCV 2025），使用 [deep-person-reid / torchreid](https://github.com/KaiyangZhou/deep-person-reid)。使用时请引用原方法和相关数据集。第三方代码版权和许可保留各自归属，详见 [来源说明](THIRD_PARTY_NOTICES.md)。本仓库暂不额外声明统一开源许可证。
