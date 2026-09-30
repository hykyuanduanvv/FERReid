# 架构与实验协议

## 模型数据流

训练每个 batch 等概率选择一个源数据集，在该域有放回抽 64 个身份条目，每个身份有放回抽 2 张图像，得到 `(64,2,3,256,128)`。因此 64 条目不保证为 64 个不同身份，同一个身份的两张图也不保证不同；训练阶段不强制跨摄像头。

冻结 ViT 副本抽取 768 维 CLS；每道题拼接两张图的特征，经两层 Q-Former 变成 32×1024 视觉嵌入，再附加一个 yes/no token。64 道题合计 2,112 个 token；追加 384 个可学习查询后输入长度为 2,496。Qwen3-0.6B 参数冻结，但训练时仍通过它对可学习输入和前端传播梯度。

384 个查询隐状态经无偏置线性映射从 1024 维变成 768 维，再整理为 `12×32×768`。ViT 每层插入 32 个 prompt，下一层替换掉上一层 prompt，保留更新后的 CLS 和图像 patches。最后输出归一化 CLS 特征。

默认支持特征来自冻结副本（`icl_feature=frozen`）。消融 `trained` 则从带 LoRA、无 prompt 的检索编码器取特征，并停止该支路梯度；它是单独消融配置。

## 损失与训练参数

`L = L_triplet + 1.0 L_ICL + 0.01 L_WPA`。

- Triplet：归一化 CLS 的 hardest positive/negative，margin=0.1；`id_loss` 名称不表示身份分类交叉熵。
- ICL：视觉对 token 的位置被 mask，只在答案位置计算自回归预测损失。
- WPA：patch 间最优传输距离，减小正对距离并增大负对距离；该损失可以为负值，不能仅由负号判定数值错误。
- 主干和 Qwen 冻结；训练 Q-Former、提示查询、prompt 映射、最后四层 QKV LoRA。
- FP16 主体、FP32 可训练参数；Adam 1e-4，恒定调度，无 weight decay、无梯度裁剪。

## 数据域划分

| 折 | 源训练域 | 留出验证域 |
|---|---|---|
| 主折 / fold 1 | Market1501 + MSMT17 | CUHK03 |
| fold 2 | Market1501 + CUHK03 | MSMT17 |
| fold 3 | MSMT17 + CUHK03 | Market1501 |

三折每折 1,000 步。单折 3,000 步扩展沿用主折。训练源域合并该域 train/query/gallery，验证域从未参与本折梯度更新；默认验证限 500 个身份，不等于全验证图库结果。

VIPeR、GRID、i-LIDS 为目标域，每个目标 split 从 train 取支持，query/gallery 作检索。测试时使用已知身份组模拟标注预算，每人取两张不同摄像头图像，组成 k 个正对；由这些图像进一步构造 64 道 yes/no 题。prompt 在同一评估条件下为整个域共享，参数不更新。

支持抽样种子为 `seed + 1000*split`；诊断/敏感度脚本另使用 `base_seed + 100000*split + 1000*k + i`，不能把不同脚本的 seed 数值直接当作同一支持集。评估使用原图与水平翻转特征相加、L2 归一化、余弦距离及 torchreid 默认排名协议（排除相同身份相同摄像头项）。

## 现有选择与诊断

`first` 取符合跨摄像头条件的前 k 个身份；`random` 无放回随机取 k 个身份。当前候选池用真实 PID 分组，且以真实跨摄像头身份是否存在筛选。因此它是按已知身份组模拟标注的框架，尚不能宣称完整解决“未标注池上无标签选择身份”的问题。

诊断包括 true/shuffle/flip/all_yes/all_no/cross/source/pseudo/zero。`shuffle` 在固定图像对上置换答案，保持答案数量，但没有保证每个答案都变化；`flip` 全部反转；`pseudo` 以图像增强副本组成伪正对，不保证跨所选图片的身份关系一定正确。

`context_sensitivity.py` 比较随机身份组、first、全零提示和固定组下题目抽样波动。`oracle@N` 从测试结果中挑选最好支持集，是有测试信息的探索性上界，不能作为无泄漏方法主结果。

## 可支持的结论范围

主表完整系统优于当前 plain 基线；由于参数数目、WPA、上下文训练等同时变化，不能将全部收益归因于具体样例内容。零 prompt 与无 prompt 结构也不等价。历史诊断中部分扰动差异较小，但只有聚合 CSV，没有完整逐查询排名和配对置信区间，不能声称所有查询均不敏感或已严格证明等效。

目前目标域已被用于多项探索。后续根据这些结果设计新方法时，应另留未用于方法选择的最终评估划分/数据域，并进行多训练种子对照。

## 2026-10-01 新增：主干、强基线选项与 Protocol-2

**主干**（`--backbone`）：`vit_b16`（历史默认）；`dinov2_b14`——DINOv2 ViT-B/14，数据管线仍输出 256×128，模型内部缩放到 252×126（18×9 patches，最接近 2:1 的 14 倍数），位置编码由 DINOv2 从 37×37 插值。VICP 的出题副本 `encoder_copy` 与检索编码器使用同一主干。

**模型**（`--model_type`）：`vicp`；`plain`（同主干，无 prompt）；`vpt`（与 VICP 相同的 LoRA、prompt 插入与损失，prompt 为一个零初始化的可学习参数，不用 LLM 与上下文）。

**强基线选项**（默认全部关闭，等于历史模型）：
- BNNeck + ID 交叉熵（`--bnneck True --ce_loss_weight 1.0`，label smoothing 0.1）：triplet 仍作用于 L2 归一化的原始 CLS；CE 作用于 BN 后特征；开启 BNNeck 时检索用 BN 后特征。分类器覆盖全部源域身份。CE 只在训练模式计算（测试时上下文前向中的标签是图像对编号，不是源域类别）。
- 编码器训练：`--train_backbone lora`（`--lora_layers`、`--lora_rank`）或 `full`（全部编码器参数，学习率 × `--backbone_lr_mult`）。
- 采样：每个身份 K 张（`--instances_per_id`，须为偶数：VICP 出题把相邻两张当作正对），`--cross_camera_instances` 保证前两张来自不同摄像头，`--unique_ids_per_batch` 使同一 batch 不重复身份，`--batch_domain_mode mixed` 让 batch 混合多个源域。
- `--ot_loss_weight 0` 关闭 WPA 并跳过其计算；`--triplet_margin`。

**选择单位**（`--selection_unit`，默认 `image`）：见 [运行计划 §4](RUN_PLAN_20261001.md)。`identity` 为历史实现，与旧代码逐项一致（`tests/test_selection.py`）。

**Protocol-2**：Market1501、MSMT17、CUHK-SYSU、CUHK03 留一交叉；源域只用 train 部分（`--source_all_images False`），目标域的 train 部分只作无标签候选池，query/gallery 评测。调参在"目标 = CUHK-SYSU"一折内进行：Market + MSMT17 训练、CUHK03 验证。任务清单见 `plans/`。
