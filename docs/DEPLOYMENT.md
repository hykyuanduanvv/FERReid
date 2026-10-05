# 实验部署流程

本流程面向 Linux/NVIDIA 单卡。代码来自已完成实验的 AutoDL 环境；Windows 可阅读结果、运行仓库核验，完整模型训练请使用 Linux。原环境 Python 3.12.3、RTX 4090 D 24GB、torch 2.6.0 + torchvision 0.21.0；依赖版本来自实际安装记录。

## 1. 获取仓库并建立环境

私有仓库需要先在本机完成 GitHub 登录，并拥有仓库权限。

```bash
git clone -b Active_Pair_Prompt https://github.com/hykyuanduanvv/FERReid.git
cd FERReid
conda create -n ferreid python=3.12 -y
conda activate ferreid
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
mkdir -p external
git clone https://github.com/KaiyangZhou/deep-person-reid.git external/deep-person-reid
git -C external/deep-person-reid checkout f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
```

本项目使用源码中的 `torchreid` 包，通过 `PYTHONPATH` 加载。不要以同名 PyPI 包替代。**请编译 Cython 排序评估**：三个目标域都较大，尤其 MSMT17（11,659 query × 82,161 gallery），Python 版会非常慢：

```bash
(cd external/deep-person-reid/torchreid/metrics/rank_cylib && python setup.py build_ext --inplace)
```

编译后 `evaluate_rank` 自动使用 Cython 版本（结果与 Python 版相同，可用同目录的 `test_cython.py` 核对）。DINOv2 主干还需要其模型代码：`git clone https://github.com/facebookresearch/dinov2.git external/dinov2`，并设置 `FERREID_DINOV2_REPO`。`requirements.txt` 已列入主流程和诊断的依赖（原 Conda 环境快照保留在 Direction_A 分支的 `env/`）。

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
└── cuhksysu/cuhksysu4reid/          # 三折的源域；读取器 adapters/cuhksysu.py
    ├── train/   ├── query/   └── gallery/     # <pid>_*.jpg 或 <pid>/*.jpg
```

CUHK-SYSU 使用行人搜索数据集裁剪出的 ReID 版本（DG-ReID 留一协议所用划分）。它没有摄像头标签：读取器令 train/query 的 camid 为 0、gallery 为 1，使"同人同摄像头"过滤不删除真实匹配；标注模拟对它不要求跨摄像头。文献中的规模为 train 5,532 人 / 15,088 张、query 2,900、gallery 5,447（2,900 人），**放好数据后以 `python scripts/check_datasets.py --domains cuhksysu` 的输出为准**，不符时先核对版本与目录再使用。

CUHK03 需使用 **NP labeled 的 767/700 身份划分**，文件名格式为 `<camera_pair>_<pid>_<camera>_<index>.jpg`；不要把原始 `.mat`、detected 版本或其他协议放进相同目录后直接套用本表。

数据图片需自行从发布方获得并遵守原协议。放好图片后执行：

```bash
python scripts/check_datasets.py
```

它按训练和评测的方式读取四个数据集，打印规模、摄像头数和路径泄漏检查；退出码不代表所有数据均成功，应检查打印的 `Summary`。当前检查仅覆盖统计和路径交集，并未提供内容级无泄漏保证。

## 4. 准备模型权重

ViT 的文件名必须是：

```text
$FERREID_WEIGHTS_DIR/vit_base_patch16_224.pth
```

优先使用原实验保留的权重（SHA256 记录在 Direction_A 分支的 `results/archive_20260929/provenance.json`）。新的部署也可从具名 timm 预训练权重准备：

```bash
python scripts/prepare_weights.py --model vit_b16 --output-dir "$FERREID_WEIGHTS_DIR"
# DINOv2 ViT-B/14（--backbone dinov2_b14）：官方权重 -> dinov2_vitb14_pretrain.pth
python scripts/prepare_weights.py --model dinov2_b14 --output-dir "$FERREID_WEIGHTS_DIR"
#   无法联网时复制已有文件：--from-file ~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
```

9/30 实验所用 DINOv2 权重的 SHA256：`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。缺少任一主干权重时代码直接报错，不会静默随机初始化。

脚本加载 `vit_base_patch16_224.augreg2_in21k_ft_in1k`，将位置编码适配到 256×128（16×8 patches），移除分类头，并检查完整 state_dict 可严格加载。它拒绝覆盖已有权重。新下载和重新序列化的文件不保证与历史权重逐字节一致；精确复算旧 checkpoint 应使用历史原文件。

准备 Qwen3-0.6B（只有 VICP 对照需要）：

```bash
hf download Qwen/Qwen3-0.6B --local-dir "$QWEN3_06B_DIR"
```

该目录应含模型配置、分词器及权重。记录所下载的模型 revision；下载当前默认分支不能视为历史 revision 的严格复现。离线运行时先确认文件齐全，再设置 `HF_HUB_OFFLINE=1`。

历史训练 checkpoint 未上传到 Git。Direction_A 分支训练的 VPT / plain / 原版 VICP checkpoint 在本分支可直接加载（参数名未变）；残差 prompt 等方向 A 的 VICP 变体不能加载。

## 5. 训练基础模型

所有实验通过多卡启动器运行：一张卡一个任务，自动排队，日志在 `experiments/_launch/<任务名>.log`，成功的任务再次运行时自动跳过。

```bash
python scripts/launch_tasks.py plans/base.tasks --gpus 0,1,2,3 --dry-run   # 先查看命令
python scripts/launch_tasks.py plans/base.tasks --gpus 0,1,2,3
```

`plans/base.tasks` 为三折各训练三个模型：主模型 `base_md_<目标>`（VPT + 每个源域 8 个追加 token）、消融 `base_vpt_<目标>`（只有一组共享 prompt）、in-context 对照 `base_vicp_<目标>`。源域只用 train 划分，不做验证，12,000 步。损失默认 BNNeck + ID 交叉熵 + triplet + WPA，可在 `plans/chosen.sh` 里用 `LOSS_ARGS` 覆盖。

## 6. 主动 pair 查询实验

```bash
python scripts/launch_tasks.py plans/tune.tasks --gpus 0,1,2,3     # 在 CUHK-SYSU 上选 lr / 步数，结果写入 plans/active_chosen.sh
python scripts/launch_tasks.py plans/diag.tasks --gpus 0,1,2       # 诊断
python scripts/launch_tasks.py plans/active.tasks --gpus 0,1,2,3   # 主表、消融、in-context 对照
```

方法、协议、输出与判读见 [ACTIVE_PROMPT.md](ACTIVE_PROMPT.md)。

## 7. 自检

```bash
python tests/test_active.py          # 主动模块：约束、候选、策略、端到端（CPU，约 1 分钟，无需数据和权重）
python tests/test_trainer.py         # 多域 token 训练：经 HF Trainer 的单域 batch、梯度只到本域 token、checkpoint 重载
python tests/test_models.py --tiny   # 模型前后向（随机小 ViT）
python tests/test_selectors.py       # 锚点图选择器（合成数据）
python tests/test_models.py --skip_vicp   # 真实主干权重（需第 4 节的权重）
```

保存完整训练参数、训练种子、数据划分和 CSV。`seed` 只标识一次训练，不能代替多个训练种子。
