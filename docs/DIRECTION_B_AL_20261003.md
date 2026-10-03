# Direction_B_AL：四折七方法短程主动选样微调

本分支归档 **2026-10-03 03:14 完成的 168 组短程微调实验**。目标域选样后更新固定 VPT Prompt，属于有监督目标适配；它不是后续 P3 对比或目标域 training-free 的 VICP 实验。

完整记录位于 [`results/Direction_B_AL_20261003`](../results/Direction_B_AL_20261003/README.md)。原始训练代码、参数与结果保持原样，本次上传没有重新训练或按测试成绩更换检查点。

## 固定实验设计

| 项目 | 设置 |
|---|---|
| 目标域 | Market1501、CUHK-SYSU、CUHK03、MSMT17，四折分别训练源模型 |
| 七种选样方法 | Random、Dedup、K-center、Typical、Hard Negative、Facility、Facility Camera |
| 重复方式 | 每方法每折 3 次支持抽样 × 2 个优化种子（42、43）；共 168 组 |
| 源模型 | DINOv2-B/14 + LoRA（最后4层QKV，rank128）+ 固定VPT，每层32 token |
| 源训练 | 其余三域的 train 划分，12,000步；Triplet + 0.01 WPA；不使用目标域测试集选择检查点 |
| 适配初始化 | 该折源模型 checkpoint-12000 的 Prompt；各组独立从该 Prompt 开始 |
| 适配参数 | 只更新外部 Prompt，主干与 LoRA 冻结，模型 eval 模式下启用 Prompt 梯度 |
| 标注预算 | k=16 个图像 anchor；实际有效身份数以每组 selection 记录为准，不保证16个不同身份 |
| 适配损失 | FP32 hardest Triplet，margin=0.1；目标适配不加WPA |
| 优化器 | Adam，lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 更新约束 | 每步投影至源 Prompt 的相对 L2 改变量≤10% |
| 步数 | 每组100次成功更新；保存0/10/30/50/100步，完整评估30/100步 |
| 主结果 | 预定100步；30步为次要分析，不按目标测试分数挑选最优步数 |

选样先在匿名特征与可用相机信息上进行，再通过固定标注随机数构建支持关系。7种方法是既有选样启发式，本实验不是逐轮追加标注的主动学习循环。

## 如何查看与核验

结果表、原始完成状态和文件索引见[归档首页](../results/Direction_B_AL_20261003/README.md)。无需数据集、PyTorch或GPU即可复算并核验归档：

```bash
python scripts/audit_direction_b_al_archive.py
```

此命令验证所有归档文件的SHA256、原始代码哈希、168组各100条训练日志、30/100步指标汇总、Prompt更新约束以及四折独立复核记录。它核验已保存的实验记录，不重新提取图像特征，也不冒充一次新的GPU复现。

原实验执行入口为归档中的 `run/code/runner.py`。该文件按原始字节保留，内部运行根目录取其父目录、仓库路径固定为 `/root/hyk/FERReid`；**不要在只含文本记录的归档目录直接启动训练控制器**。

原服务器保留完整数据和二进制检查点，可独立重评已有实验，例如：

```bash
cd /root/hyk/FERReid
CUDA_VISIBLE_DEVICES=<空闲卡号> /SSD_Data01/miniforge/envs/ferreid/bin/python \
  experiments/p2b_short_20261003_v1/code/runner.py --stage verify --domain market1501
```

其他目标域同理。此命令会重新评估源模型和Random/Facility的固定检查点，并更新原运行的verify记录；本次发布没有执行它，因为已有四折独立重评均通过。

迁移机器时，应先按仓库部署文档准备环境与数据，恢复 `dependencies/p2b_20261002_v1` 中的记录及 `dependencies/code_snapshot` 中的原代码，并取得文件索引中列出的权重。原始绝对路径参与部分数据及选样哈希，不能仅更改路径就声称完成了精确重放。超过1MiB的JSON采用gzip无损压缩，解压后可恢复原文件及哈希。

## 结果解释边界

- 所有主结果均为预定100步的6组均值；只有3次独立支持抽样，不能把6组都当作独立数据抽样。
- Hard Negative的四折平均最高，约65.42；基线约64.58，Random约64.69。各目标域的最优方法不同。
- 七种方法均表现为30步四折均值高于100步，但主结果仍报告100步，不据此追选有利检查点。
- MSMT17沿用用户接受的原始划分，保留已发现的214组跨train/test完全相同内容；数据例外记录随归档提供。
- 原有Prompt token对称性保留；本实验没有声称已修复该结构问题。
- 结果属于修订后的探索性协议：一个源训练种子，CUHK03曾参与历史方法开发。保留源配置中的相关说明。

## 未放入Git的文件

数据集图像、源模型权重及约1GB的Prompt检查点保留在原服务器。`archive_manifest.json` 列出856个服务器端二进制文件的路径、大小和SHA256，Git中保留全部本次运行的文本记录。历史5000步目标微调结果及后续P3新实验不属于本次新增归档。
