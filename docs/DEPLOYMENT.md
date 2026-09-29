# 实验部署流程

本流程面向 Linux/NVIDIA 单卡。代码来自已完成实验的 AutoDL 环境；Windows 可阅读结果、运行仓库核验，完整模型训练请使用 Linux。原环境 Python 3.12.3、RTX 4090 D 24GB、torch 2.6.0 + torchvision 0.21.0；依赖版本来自实际安装记录。

## 1. 获取仓库并建立环境

私有仓库需要先在本机完成 GitHub 登录，并拥有仓库权限。

```bash
git clone https://github.com/hykyuanduanvv/CVPR-2027.git
cd CVPR-2027
conda create -n ferreid python=3.12 -y
conda activate ferreid
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
mkdir -p external
git clone https://github.com/KaiyangZhou/deep-person-reid.git external/deep-person-reid
git -C external/deep-person-reid checkout f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
```

本项目使用源码中的 `torchreid` 包，通过 `PYTHONPATH` 加载。不要以同名 PyPI 包替代。无需编译 Cython，未编译时使用 Python 排序评估。`requirements.txt` 已列入主流程和诊断的依赖；`env/requirements_frozen.txt` 含原 Conda 环境的本机 `file://` 包，仅供追溯，**不要对该快照执行 pip install -r**。

固定版本反映原服务器实况，并不代表已验证所有软件镜像都能下载。若某镜像缺包，先检查正确包源；不要无记录地升级关键依赖。`CustomTrainer` 使用 transformers 的内部接口，版本变动可能导致签名不兼容。

## 2. 配置路径

```bash
cp configs/paths.example.sh configs/local.sh
# 编辑文件：数据目录、权重目录、Qwen 目录和 Python 解释器
source configs/local.sh
python -c "import torch, timm, transformers, torchreid; print(torch.__version__, torch.cuda.is_available(), timm.__version__, transformers.__version__)"
```

环境变量如下；路径应使用绝对路径。

| 变量 | 用途 | 未设置时的历史默认值 |
|---|---|---|
| `FERREID_DATA_ROOT` | 数据集根目录 | `/root/autodl-tmp/reid-data` |
| `FERREID_WEIGHTS_DIR` | ViT 权重目录 | `/root/autodl-tmp/weights` |
| `QWEN3_06B_DIR` | 本地 Qwen 模型目录 | `/root/basic-models/Qwen3-0.6B` |
| `PYTHON` | shell 脚本使用的解释器 | 当前 `python` |
| `PYTHONPATH` | 固定版本 torchreid 源码 | 配置示例指向 `external/deep-person-reid` |

数据和权重建议放在 AutoDL 数据盘；Git 仓库只保存代码和小型结果文件。`configs/local.sh` 被忽略，不要把访问凭据写进代码。

## 3. 准备数据和精确划分

数据集版本、规模及角色见 [README 数据集表](../README.md#数据集)。目录必须匹配读取器预期：

```text
$FERREID_DATA_ROOT/
├── market1501/Market-1501-v15.09.15/
│   ├── bounding_box_train/  ├── query/  └── bounding_box_test/
├── msmt17/MSMT17_V1/            # 包含发布包的图像与 list_*.txt
├── cuhk03/CUHK03_pytorch/CUHK03_pytorch/
│   ├── train_all/<pid>/*.jpg
│   ├── query/<pid>/*.jpg
│   └── gallery/<pid>/*.jpg
├── viper/VIPeR/{cam_a,cam_b}/
├── grid/underground_reid/{probe,gallery}/
└── ilids/i-LIDS_Pedestrian/Persons/
```

CUHK03 需使用 **NP labeled 的 767/700 身份划分**，文件名格式为 `<camera_pair>_<pid>_<camera>_<index>.jpg`；不要把原始 `.mat`、detected 版本或其他协议放进相同目录后直接套用本表。

数据图片需自行从发布方获得并遵守原协议。本仓库只提供历史划分元数据。先放好图片，再执行：

```bash
# 默认只核对图像是否齐全、已存在 splits.json 是否一致
python scripts/install_splits.py
# 只创建缺失的 splits.json；遇到不同的现有划分会停止，不自动覆盖
python scripts/install_splits.py --apply
python scripts/check_datasets.py
```

VIPeR/GRID 的路径会转换为当前数据根目录；i-LIDS 保留读取器需要的图像文件名。若现有划分与记录不同，应先将其另存后再有意切换，不要混用不同划分下的结果。

`check_datasets.py` 对 PRID2011 显示缺失是当前快照的预期状态。其退出码不代表所有数据均成功，应检查打印的 `Summary`。正式命令显式选择 `viper,grid,ilids`。当前检查仅覆盖统计和路径交集，并未提供内容级无泄漏保证。

## 4. 准备模型权重

ViT 的文件名必须是：

```text
$FERREID_WEIGHTS_DIR/vit_base_patch16_224.pth
```

优先使用原实验保留的权重；其 SHA256 记录在 `results/archive_20260929/provenance.json`。新的部署也可从具名 timm 预训练权重准备：

```bash
python scripts/prepare_weights.py --output-dir "$FERREID_WEIGHTS_DIR"
```

脚本加载 `vit_base_patch16_224.augreg2_in21k_ft_in1k`，将位置编码适配到 256×128（16×8 patches），移除分类头，并检查完整 state_dict 可严格加载。它拒绝覆盖已有权重。新下载和重新序列化的文件不保证与历史权重逐字节一致；精确复算旧 checkpoint 应使用历史原文件。

准备 Qwen3-0.6B：

```bash
hf download Qwen/Qwen3-0.6B --local-dir "$QWEN3_06B_DIR"
```

该目录应含模型配置、分词器及权重。记录所下载的模型 revision；下载当前默认分支不能视为历史 revision 的严格复现。离线运行时先确认文件齐全，再设置 `HF_HUB_OFFLINE=1`。

历史训练 checkpoint 未上传到 Git；服务器原路径与文件指纹见 `provenance.json`。仅有代码仓库不能直接进行历史权重推理，需要已有 checkpoint 或重新训练。

## 5. 运行训练

```bash
# Market+MSMT → CUHK03；1,000 steps，seed=42
bash scripts/run_main.sh vicp main_vicp 1000 42
bash scripts/run_main.sh plain main_plain 1000 42

# 三折源域交叉验证；每折 1,000 steps
bash scripts/run_cv.sh cv_new 1000
python scripts/summarize_cv.py cv_new

# 单折延长到 3,000 steps（新的实验名称避免覆盖）
bash scripts/run_main.sh vicp main_vicp_3k 3000 42
```

核心参数：batch=64 身份采样条目，每条目 2 图；Adam，lr=1e-4，恒定学习率，weight decay=0，不裁剪梯度，FP16；每 100 步在固定 CUHK03 **500 个测试身份子集**验证。原采样器按人工构造的大步数流工作，日志中 `epoch=0.01` 等不能当成真实整数据集遍历次数。

`run_main.sh` 拒绝使用已有输出目录。原兼容脚本 `run_cv_fold1_3k.sh`、`run_v1.sh`、`run_v1_diag.sh` 保留固定输出名，使用前检查已有结果。`run_v1.sh` 是 `icl_feature=trained` 的消融，不属于主表配置。

模型只在最后保存 checkpoint。某个中间步的验证值最好，不表示对应权重已保存。继续训练需明确使用 `scripts/train_reid.py --resume_from_checkpoint <目录>` 并提供一致参数。

后台执行可用：

```bash
mkdir -p experiments
nohup bash scripts/run_main.sh vicp main_vicp_bg 1000 42 > experiments/main_vicp_bg.launcher.log 2>&1 &
```

此部署脚本不包含自动关机命令。租赁机器应在确认实验结束、结果已备份后按平台流程关机，避免仍有任务时被通用脚本意外关停。

## 6. 目标域正式评测

```bash
python scripts/eval_context.py \
  --output_dir experiments/main_vicp/eval \
  --checkpoint experiments/main_vicp/val_cuhk03/checkpoint-1000 \
  --model_type vicp --domains viper,grid,ilids \
  --methods random --ks 16 --eval_seeds 3 --eval_splits 10 \
  --num_icl_samples 64 --fp16 True --report_to none

python scripts/eval_context.py \
  --output_dir experiments/main_plain/eval \
  --checkpoint experiments/main_plain/val_cuhk03/checkpoint-1000 \
  --model_type plain --domains viper,grid,ilids \
  --methods random --ks 16 --eval_seeds 1 --eval_splits 10 \
  --num_icl_samples 64 --fp16 True --report_to none
```

每项结果进入 `context_eval.csv`。评估入口为兼容原 Trainer 会先读取源训练数据，因此正式评估也需准备 Market/MSMT。必须确认日志为 `missing: 0 unexpected: 0`，且三个域均成功、有完整行数（VICP 90 行，plain 30 行）；旧入口遇到数据缺失会跳过域，不能只看进程退出码。

## 7. 上下文诊断

```bash
# 正确、乱序标签、反转标签、全 yes/no、跨域、源域、伪支持和零 prompt
python scripts/context_diagnostics.py \
  --output_dir experiments/main_vicp/diagnostics \
  --checkpoint experiments/main_vicp/val_cuhk03/checkpoint-1000 \
  --domains viper,grid,ilids --ks 4,16 --n_contexts 8 --eval_splits 3 \
  --num_icl_samples 64 --fp16 True --report_to none

# 更换支持身份与固定支持下的问题抽样波动
python scripts/context_sensitivity.py \
  --output_dir experiments/main_vicp/sensitivity \
  --checkpoint experiments/main_vicp/val_cuhk03/checkpoint-1000 \
  --domains viper,grid,ilids --ks 2,4,8,16,32 --n_contexts 30 --noise_reps 5 \
  --eval_splits 1 --num_icl_samples 64 --fp16 True --report_to none
```

诊断会缓存图像到 GPU，规模大于目前小目标集时需要调整缓存实现。`noise` 表示固定图像对、改变题目抽样种子，不是随机 prompt 向量；`zero` 是插入全零 token，不等于完全移除 token。`oracle` 使用测试成绩事后选优，仅能作探索性上界。

## 8. 验证与归档

```bash
python scripts/verify_repository.py
python scripts/summarize_results.py
python -m compileall -q adapters ops scripts models.py custom_trainer.py
```

保存完整训练参数、训练种子、支持种子、数据划分、checkpoint 指纹和 CSV。`seed=42` 只标识历史一次训练，不能代替多个训练种子；支持种子重复也不能代替训练重复。当前代码的模型构造早于 Trainer 设置随机种子，故从头重训不保证完全确定；这一继承行为在本次发布中未改变。进一步严格重训应在模型构造前统一设种子，并作为新协议单独记录。

本次发布前的实际检查范围见 [复现记录](REPRODUCIBILITY.md)；没有在空白 Conda 环境完整重装并重新训练，因此不声称已完成端到端从零复现。
