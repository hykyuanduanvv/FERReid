# 发布整理与复现记录

来源：2026-09-29 服务器 `/root/FERReID` 代码、`FERReID_experiments` 中的 CSV/训练状态、目标域划分。主模型计算代码、损失和训练数据抽样逻辑保持服务器快照语义；本次未在原服务器源码目录编辑文件，也未重新训练。

## 发布版变更

1. `FERREID_DATA_ROOT`、`FERREID_WEIGHTS_DIR` 可配置，保留 AutoDL 历史默认值。
2. 缺少 ViT 预训练权重时明确失败，防止旧路径静默随机初始化。
3. shell 脚本使用当前仓库位置和可配置 Python，传播失败退出码；新增主折启动脚本，拒绝覆盖已有实验目录。
4. 补齐诊断与 torchreid 依赖，增加具名权重准备和记录划分恢复工具。
5. 新增中文说明、历史数据指纹、无需 GPU 的结果汇总与完整性核验。

## 可复核证据

`provenance.json` 记录的是**整理前原代码**和历史结果的指纹，并非声称发布版所有文件与原树逐字节一致。结果文件按原字节归档。历史 checkpoint 指纹可供拥有对应文件的研究者核对；大权重与图像不在仓库内。

2026-09-29 发布前检查结果：

| 检查 | 结果 |
|---|---|
| 历史结果 CSV/JSON 指纹 | 14 个文件 SHA256 全部一致 |
| Python 语法 | 24 个文件通过解析 |
| 目标域划分 | 40 个记录划分结构/路径交集检查通过；恢复到服务器根目录后与既有 splits.json 全部一致 |
| 主表 | 从 30 行 plain 和 90 行 VICP 原始 CSV 重算，与 README 一致 |
| shell | 所有 `scripts/*.sh` 通过 `bash -n` |
| 命令入口 | train_reid / eval_context / context_diagnostics / context_sensitivity 的 `--help` 成功 |
| checkpoint 加载 | 发布版在临时目录严格加载主折 checkpoint-1000 成功，未修改原工作树 |
| GPU 前向 | VIPeR split 0 的 16 个支持身份生成 `(1,12,32,768)` prompt；两张 query 输出 `(2,768)` 特征，数值有限且范数接近 1 |

GPU 检查只做冻结模型前向，没有优化器更新。原源码保留的非执行 AdaFace 字符串转义、WPA AMP 旧接口和未编译 Cython 会打印 warning，不影响上述检查。没有在全新环境重新安装全部依赖、训练完整模型、复跑全部目标域实验，不能称为完整从零复现。

## 已知限制

- 目标域路径交集为零不等于文件内容无重复；当前未重新审计行人数据的全部字节哈希或近重复。
- 模型在 Trainer 统一设种子之前构造；历史 seed=42 不能保证从头训练的初始化逐位可复现。后续改进需作为新协议保存，不能冒充原始运行。
- 原 `CustomTrainer` 保留被 DGReIDTrainer 覆盖的其他任务旧方法；只能使用 `scripts/train_reid.py` 等行人入口，不能直接把父类当作独立训练入口。
- 原评估入口允许 `strict=False` 加载，需检查日志 missing/unexpected 均为 0；不要忽视结构或参数不匹配。
- 源域 combineall、目标域少量标注、500 身份验证子集都属于实验协议的一部分，与其他论文比较时必须说明。
- 选择器当前使用已知身份分组；first/random 不等于已实现无标签主动选择算法。
- 支持身份种子、上下文题目种子与训练种子是不同随机来源。现有结果不足以估计训练随机性。
- 历史结果无法追溯到独立的上游 Qwen revision；优先保留原模型文件，后续新部署应记录下载 revision 与所有输入指纹。

## 2026-10-01 发布

来源：2026-09-30 服务器 `/root/FERReID`（包含 DINOv2 主干、VPT 基线、`oracle_prompt.py`、`label_value.py`、`prompt_select.py`、`analyze_0930.py`），合并 9/29 发布版的可移植改动（路径环境变量、缺权重报错）。9/30 结果归档于 `results/archive_20260930/`（CSV、训练状态、图与报告；不含 checkpoint）。

计算相关的修改（均不改变默认配置下的结果）：
1. LoRA 位置 / 秩、triplet margin、WPA 权重 0 时跳过计算、BNNeck/CE 头、全参数微调均为可选项，默认值即历史配置。
2. `models.py` 中 `input_ids` 与 WPA 标签由 `.cuda()` 改为跟随输入设备；`ops/losses.py` 的 triplet 掩码由"有 CUDA 就放 cuda"改为跟随标签设备（GPU 上行为不变，修复 CPU 张量在 CUDA 机器上的设备不一致）。
3. 目标域距离矩阵改为分块计算（GPU 可用时在 GPU 上），数值与原 CPU 计算一致到浮点误差。
4. 评测从 checkpoint 的 `training_args.bin` 读取结构参数；权重键不匹配时报错。
5. **`--selection_unit` 默认 `image`**：历史脚本（`scripts/run_*.sh`）与文档中的历史命令已显式加 `--selection_unit identity`。

2026-10-01 发布前检查（服务器 CPU，无 GPU 训练）：

| 检查 | 结果 |
|---|---|
| `python -m compileall`、`bash -n` 全部脚本 | 通过 |
| `tests/test_selection.py --legacy <9/30 服务器 context_selection.py>` | 5 个数据集、k=2/4/16、各 5 个种子：`identity` 与旧实现逐项一致；`image` 的配对均为同一人、跨摄像头（有摄像头的域）、每人至多一次、`k = n_pairs + n_fail + n_dup` |
| `tests/test_models.py` | plain / VPT / VICP × ViT / DINOv2 × BNNeck/CE、全参、LoRA 8 层 r=32、K=4：前后向通过，默认配置参数量与历史一致（plain 2.36M、VICP 37.95M） |
| 端到端（CPU：`launch_tasks.py` → `plans/common.sh` → VPT DINOv2 + BNNeck/CE + K=4 跨摄像头采样 + cosine，训练 4 步 → CUHK03 验证 → checkpoint → `eval_context.py` 读回结构评测 VIPeR） | 两个任务状态 0；评测自动恢复 `vpt / dinov2_b14 / bnneck / num_train_ids=1792`（Market 751 + MSMT17 1,041，仅 train 部分），missing/unexpected 均为 0 |
| 回归：新代码评测 9/30 的 VICP DINOv2 checkpoint（VIPeR split 0，`identity`，k=16，种子 0） | mAP 84.10 / R1 77.85（CPU fp32），存档 84.00 / 77.53（GPU fp16）；差异在精度范围内（R1 相差 1/316 个 query），上下文选择完全一致 |
| `scripts/verify_repository.py` | 通过（9/29 归档 14 个哈希不变） |
| 任务清单 `--dry-run` | 4 个阶段共 42 个任务 |
| DINOv2 权重 | `prepare_weights.py --model dinov2_b14 --from-file` 严格加载通过 |
| Cython 排序 | 在临时副本中 `setup.py build_ext --inplace` 编译成功 |

未做：GPU 上的完整训练与 Protocol-2 评测（交由运行计划执行）；CUHK-SYSU 数据尚未取得，读取器未在真实数据上验证。

### 方向 A 代码检查（2026-10-01，服务器 CPU）

| 检查 | 结果 |
|---|---|
| `tests/test_models.py` | 原 6 个变体数值与加入方向 A 代码前完全相同（默认不变）；残差 + EMA + episode 变体前后向通过，所有新参数有梯度，EMA 均值被更新，两个不同上下文生成的 prompt 余弦 0.23（< 1，prompt 随上下文变化） |
| 摄像头对伪域（调参折 Market + MSMT17 train 部分） | `--pseudo_min_ids 64`：62 个伪域（Market 15、MSMT17 47），14,807 个样本，覆盖 1,771 / 1,792 人 |
| 端到端（`launch_tasks.py`：VICP DINOv2 + BNNeck/CE + 残差/EMA + 摄像头对伪域 + episode，训练 4 步 → `context_gain.py`（VIPeR、i-LIDS）→ `eval_context.py`） | 三个任务状态 0；评测从 checkpoint 恢复 `prompt_mode / ctx_center / pseudo_domains / num_train_ids=1771`，missing/unexpected 均为 0。4 步模型的指标无意义，只验证流程 |

### 方向 B（选择算法）代码检查（2026-10-01，服务器 RTX 4090 D）

| 检查 | 结果 |
|---|---|
| `tests/test_selectors.py`（合成数据） | 11 个选择器均返回 k 个不同的合法索引、同种子可复现；**隐藏标签被访问即报错的守卫未触发**；去重类选择器无重复的人；`camera_balanced` / `facility_camera` 覆盖全部 4 个摄像头；`typical` 典型性最高、`kcenter` 多样性最高 |
| `tests/test_selectors.py --real`（VIPeR / GRID split 0，k=16，10 个种子） | 用 9/30 训练好的 VPT DINOv2 特征：多数选择器得到 15.6–16.0 个有效配对；`first` 在 VIPeR 只有 8 个（相邻两张是同一人）。用未训练的预训练特征时 GRID 上 `facility` / `typical` 会重复选同一人（1.4–2.0 张），因此**选择特征默认改用模型训练过的编码器（无 prompt）** |
| 回归 | `tests/test_models.py` 数值与之前完全一致；`tests/test_selection.py` 的 `identity` 仍与旧实现逐项一致 |
| GPU 冒烟（launcher） | `eval_selectors.py` 两种生成器（tuned / incontext）、`eval_context.py --methods random,typical,facility_camera`（自动提取候选池特征并记录选择性质）、`--train_context_selector typical` 训练：全部成功 |
| 发现并修复 | BNNeck 头在 fp16 训练中崩溃（BN 缓冲区与冻结 bias 仍为 fp16，旧代码只检查已转为 fp32 的 weight）；CPU（fp32）测试无法发现，GPU 测试暴露 |
| 真实配置的显存与速度（batch 64，fp16，24 GB） | 全参微调 3.0 步/秒；方向 A 全开 1.35 步/秒；VPT + BNNeck 2.9 步/秒；方向 A + 训练时 `facility` 选择 1.21 步/秒；峰值显存 21.4 GB |
