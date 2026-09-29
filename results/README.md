# 历史实验结果

`archive_20260929/` 从原服务器数据盘导出，保存逐次 CSV 和非 checkpoint 目录中的 `trainer_state.json`；未上传图像、大模型权重或私有凭据。`provenance.json` 记录原源码、结果文件和权重的 SHA256。归档目录禁用 Git 换行转换以保留字节指纹。

## 主表

| 目标域 | plain mAP / Rank-1 | VICP mAP / Rank-1 |
|---|---:|---:|
| VIPeR | 51.83 / 41.17 | 59.21 / 49.62 |
| GRID | 35.89 / 26.88 | 43.99 / 35.17 |
| i-LIDS | 69.68 / 58.83 | 76.07 / 67.17 |

计算入口：`python scripts/summarize_results.py`（从仓库根目录运行）。

| 模型 | checkpoint 原相对路径 | 原始 CSV |
|---|---|---|
| plain | `baseline_plain/val_cuhk03/checkpoint-1000` | [30 行结果](archive_20260929/baseline_plain/eval_plain/context_eval.csv) |
| VICP | `cv_0926/val_cuhk03/checkpoint-1000` | [90 行结果](archive_20260929/baseline_plain/eval_vicp_fold1/context_eval.csv) |

路径前缀为原服务器 `/root/autodl-tmp/FERReID_experiments/`；权重没有放在此仓库中。CSV 位于 `baseline_plain/` 下是历史组织方式，不表示两份结果使用同一个模型。

两个模型均在 Market+MSMT（combineall）训练 1,000 步。VICP：k=16，random，10 个划分，支持种子 0/1/2；plain：同样 10 个划分、1 次评估，忽略 prompt。旧脚本先对每个种子的 10 个划分求平均，再对种子均值计算均值与总体标准差（ddof=0）。plain 只有一个种子，历史打印的 `±0.00` 不表示跨划分没有波动；新汇总脚本标为“未估计”。这些都不是训练种子置信区间。

## 其他已完成记录

| 记录目录 | 范围与状态 |
|---|---|
| `cv_0926` | 三折，各 1,000 步，训练状态完整 |
| `cv_fold1_3k` | 主折 3,000 步，保存最终权重；尚无同规格目标域主表 |
| `diag_cv0926_fold1` | 三目标域，3 个划分，k=4/16，每条件 8 个支持种子 |
| `sens_cv0926_fold1` | 三目标域，split 0，k=2/4/8/16/32，每 k 30 组支持，5 次固定组题目抽样 |
| `v1_iclfeat` | 改用训练编码器构造上下文的消融；1,000 步，诊断仅 1 个划分、4 组支持 |

CUHK03 的 3,000 步模型最终验证 mAP=33.329、Rank-1=33.700；日志中 2,900 步的更高值 mAP=34.388、Rank-1=34.900 只是中间评估点，未保存对应 checkpoint。这些数值来自固定 500 身份验证子集。

## 解读边界

plain 与 VICP 在可训练参数数量、WPA、ICL 和 prompt 结构上均不同，主表不构成单因素上下文内容消融。诊断中的 `zero` 仍注入零 token；`noise` 仅改变问题抽样；`oracle` 使用测试成绩事后挑选。当前归档不含逐查询完整排序，也没有多训练种子重复，不能仅凭均值相近断言所有上下文无效。

仓库整理保留历史模型计算路径；修复部署路径与缺权重静默初始化不产生新的论文指标。后续新实验请用独立输出目录、固定划分和完整配置，明确标记其协议与历史记录的差异。
