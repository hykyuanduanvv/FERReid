# FERReID：主动 pair 查询的目标域 prompt 适配

**Active pair querying for target-domain prompt adaptation in person ReID**

给定一个在源域训练好的 ReID 模型（默认 DINOv2 + VPT，基础参数冻结），部署到新的摄像头网络时：模型从无标签候选池中提出"可能是同一个人"的跨摄像头图像对，人只回答"是 / 否"；正对聚成身份簇，负对成为困难负样本；据此学习一组目标域 prompt，学完冻结插入，成为该域的常参数。目标是用最少的人工回答把模型迁移到新域。

> 本分支 `Active_Pair_Prompt` 由 `Direction_A_contrast_distill` 精简而来：保留数据读取、DINOv2/ViT 主干、VPT/plain 基础模型训练、原版 VICP（in-context 对照）和锚点图选择器；新增主动查询模块。方向 A（残差 prompt、对比损失、蒸馏）、上下文诊断脚本和历史结果归档留在原分支。

## 导航

- [方法、协议、输出与判读](docs/ACTIVE_PROMPT.md)
- [环境安装、数据与权重准备、运行](docs/DEPLOYMENT.md)
- [代码来源与第三方说明](THIRD_PARTY_NOTICES.md)

## 方法概览

```mermaid
flowchart LR
    A[无标签候选池 + 摄像头标签] --> B[冻结基础模型提特征<br/>当前域 prompt]
    B --> C[跨摄像头 kNN 提出候选 pair]
    C --> D[选择策略排序<br/>cover / uncertain / ...]
    D --> E[人工回答 是/否<br/>传递性可推出的跳过]
    E --> F[正对 → 身份簇<br/>负对 → 困难负样本]
    F --> G[只训练域 prompt<br/>基础 prompt 冻结 + 追加 token]
    G -->|下一轮| B
    G --> H[冻结插入 → query/gallery 检索]
```

| 组件 | 配置 |
|---|---|
| 基础模型 | `--model_type vpt`：DINOv2 ViT-B/14（252×126）或 timm ViT-B/16；最后 4 层 LoRA r=128；每层 32 个 prompt token；triplet（+ 可选 BNNeck/ID、WPA） |
| 候选 pair | 每张图在其他摄像头中的 10 个近邻（无摄像头的域用任意其他图） |
| 选择策略 | `cover`（本方法）、`uncertain`、`balanced`、`confident`、`random`；ID 级对照 `anchor:<选择器>`；上界 `oracle_all` |
| 约束 | union-find 正簇 + 簇间 cannot-link，传递性推断 |
| 域 prompt | `append`：每层追加 8 个 token，基础 prompt 不动；`replace`：调基础 prompt 的副本；300 步，lr 3e-4 |
| 对照 | VICP（LLM + 上下文，Qwen3-0.6B）、plain、全量标注上界 |

## 数据集

下表为服务器读取器检查的单个划分统计，记为“身份数 / 图像数”。路径均相对 `FERREID_DATA_ROOT`。图片与预训练权重不放入 Git。

| 数据集 | 角色 | train / 支持候选池 | query | gallery | 数据目录与状态 |
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

目标域只用于两件事：train 划分是无标签候选池（主动查询从中提问），query/gallery 用于评测。由于用到了少量目标域人工回答，本协议是“少量主动标注的目标域适配”，不等同于不接触目标域标注的严格域泛化。

## 快速开始

```bash
cp configs/paths.example.sh configs/local.sh && source configs/local.sh   # 编辑路径后加载
python scripts/install_splits.py --apply && python scripts/check_datasets.py
python tests/test_active.py                                              # CPU 自检，无需数据和权重

python scripts/launch_tasks.py plans/base.tasks --gpus 0,1,2,3           # 训练基础模型（或复用已有 VPT checkpoint）
python scripts/launch_tasks.py plans/diag.tasks --gpus 0,1,2,3           # 诊断：基础模型能否提出有用的 pair
python scripts/launch_tasks.py plans/active_small.tasks --gpus 0,1,2,3   # VIPeR / GRID / i-LIDS 主实验与消融
python scripts/launch_tasks.py plans/active_large.tasks --gpus 0,1,2     # Market / CUHK03 / MSMT17 作目标域
```

参数见 `adapters/args_reid.py`（模型与训练）和 `scripts/eval_active.py`（主动模块）。

## 目录

```text
adapters/
  active/           主动查询模块：图像存储（缓存/流式）、候选 pair、约束、选择策略、域 prompt 训练、轮次循环
  baseline_model.py VPT / plain 基础模型，checkpoint 加载
  vicp_model.py     原版 VICP（in-context 对照）
  backbones.py      DINOv2 / timm ViT 包装
  trainer_reid.py   基础模型训练与 in-context 评测
  context_selection.py, selectors.py   锚点图选择（label-free）与模拟标注
  reid_dataset.py, cuhk03_np.py, cuhksysu.py, config_reid.py, args_reid.py, reid_head.py
ops/                LoRA、triplet、WPA、deep prompt 插入
scripts/            train_reid / eval_active / diag_retrieval / diag_vlm / eval_context / 数据与权重准备 / 多卡启动器
plans/              任务清单（base / diag / active_small / active_large）与共享设置 common.sh
tests/              CPU 自检
docs/               方法说明与部署流程
data_manifests/     VIPeR / GRID / i-LIDS 的历史划分；无图像
```

## 引用与代码来源

主干与上下文对照改编自 [VICP: Generalizable Object Re-Identification via Visual In-Context Prompting](https://arxiv.org/abs/2508.21222)（ICCV 2025），数据读取与评测使用 [deep-person-reid / torchreid](https://github.com/KaiyangZhou/deep-person-reid)。使用时请引用原方法和相关数据集。第三方代码版权和许可保留各自归属，详见 [来源说明](THIRD_PARTY_NOTICES.md)。本仓库暂不额外声明统一开源许可证。
