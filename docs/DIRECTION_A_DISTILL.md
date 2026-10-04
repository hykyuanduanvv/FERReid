# 方向 A：用域专属 VPT 监督生成器（prompt 蒸馏）

2026-10-04。接在对比损失之后（`docs/DIRECTION_A_CONTRAST.md`，结论是负面的：生成器靠"把外域 prompt 弄坏"来满足损失，而不是"把本域 prompt 调好"）。

## 思路

- **老师：** 每个训练伪域单独有一个 prompt。它从 VPT 的 prompt 出发，冻结主干和 LoRA，用该伪域全部有标注的数据调出来。
- **学生：** 上下文生成器。它看本伪域的 k 个上下文身份，生成一个 prompt；要求同一批图片在这个 prompt 下的特征和在老师 prompt 下的特征一致。
- **不会作弊：** 监督目标是一个好的 prompt，不是"比别的域好"，所以没有"把外域弄坏"这条捷径。
- **测试时：** 和原来一样，所有参数冻结，只换上下文；老师不参与测试。

## 第 1 步：可行性检查（`scripts/group_prompts.py`）

1. `--stage cluster`：把源域的摄像头单元按风格分组。
   - 摄像头单元：MSMT17 是"摄像头 × 时段"，其他数据集是单个摄像头。
   - 风格特征：每个单元用预训练 DINOv2 的 CLS 特征取均值，单元之间算余弦距离，做平均连接层次聚类。
   - 分组约束：每组至少 2 个摄像头，并且至少有 64 个身份被组内两个摄像头都拍到；不满足的组并入离它最近的组。
   - `--grouping pair` 是另一种分法：取同时被两个摄像头拍到的人数最多的 10 个摄像头对，用来对照。
2. `--stage tune`：每组调一个老师 prompt。
   - 数据：该组调优身份的全部图片，每个 batch 32 人，每人两张来自不同摄像头的图。
   - 训练：三元组损失，300 步，学习率 1e-3。
   - 留出身份按**数据集**划分（每个数据集固定 20% 的身份），任何一组的老师都不会在别组的评测身份上调过。
3. `--stage matrix`：拿每个 prompt（全局 VPT 和每个老师）去测每个组的留出身份，算 mAP 矩阵。看两个数：
   - **本组老师 − VPT：** 生成器能拿到的收益上限。
   - **本组老师 − 其他组老师：** 用错老师会掉多少。几乎不掉，就说明这些 prompt 之间没有真正的域特异性。

   **判读：** 两个差距都只有零点几个点，就不值得继续，要换成差异更大的伪域构造方式（比如风格变换合成域、加入 CUHK-SYSU）；有几个点，就进入第 2 步。

## 第 2 步：蒸馏训练（插件式，默认关闭）

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--pseudo_domains camera_group --camera_groups groups.json` | — | 训练 batch 按组划分（`CameraGroupTrainDataset`），组名是 `<dataset>\|<group>`。groups.json 里没有的数据集整体算一组 `<dataset>\|all` |
| `--prompt_teacher teachers.pt` | 空 | 第 1 步输出的老师 prompt，每个训练域都必须有 |
| `--prompt_kd_weight` | 0 | 蒸馏损失的权重，0 表示关闭，此时和原版逐位一致 |
| `--prompt_kd_mode` | rel | `rel`：让 batch 内的余弦相似度矩阵和老师一致；`feat`：每张图学生特征和老师特征的 1 − 余弦 |

- 老师那一路的特征在 `torch.no_grad()` 下计算；老师 prompt 是普通属性，不会存进 checkpoint。
- 需要 `--batch_domain_mode single`。

## CUHK-SYSU

`scripts/convert_cuhksysu.py` 从行人搜索版（`Train.mat`、`TestG100.mat`）裁出 ReID 版。核对过的数量是：train 5,532 人 / 15,088 张，query 2,900，gallery 5,447，和 DG-ReID 文献一致。它没有摄像头标签，作为一个整体的域加入时，每个人的图片按奇偶分成两半，当作"两个摄像头"。

结果与实验设置见 `results/dirA_contrast_distill_20261004/README.md`。
