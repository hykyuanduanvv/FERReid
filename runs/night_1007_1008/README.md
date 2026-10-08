# 10-07 / 10-08 实验的启动与调度脚本

这些脚本是当晚在 WHU 服务器上实际使用的队列、调度和守候脚本，原样保存，用来记录实验是怎么跑的。它们写死了服务器路径（`/data1/yangbin/dz/...`），也写死了当晚的时间窗口，**不能直接复用**；复现时参考它们的参数，改用 `scripts/launch_tasks.py` + `plans/*.tasks`。

| 脚本 | 用途 |
|---|---|
| `queue_1007.sh`、`queue2_1007.sh`～`queue4_1007.sh`、`gpu0_now_1007.sh`、`run_loop_1007.sh`、`run_oracle_1007.sh` | 10-07：出题策略的多轮循环、上限分解（`oracle_merge` / `oracle_purify`） |
| `queue_1008.sh`、`smoke_1008.sh`、`progress_1008.sh` | 10-07 晚：ε 扫描（`mix:ε`）、聚类参数扫描、收紧聚类 + 规则挑题 |
| `queue_l20.sh`、`record_l20.*` | 10-07 夜：20 轮复跑（题只在前 5 轮出），检验提问是提高上限还是只加速 |
| `queue_night.sh`、`smoke_night.sh` | 10-08 夜：E2（`oracle_merge_subset`）、BUG 开关、k1 曲线、Q4c |
| `q1_c3_driver.sh`、`e1a_*_driver.sh`、`eval_night_driver*.sh` | 逐轮重算特征（E1a）、开发半边评测（`scripts/eval_rounds_1008.py`） |
| `q4b_ms_watch.sh`、`night_fix_watch.sh`、`e2ms1_cutoff.sh`、`extend_1045.sh` | 守候与收尾脚本（占位、依赖检查、截止取消、窗口外补评测） |

结果和判定见本地文档 `02_每日任务与实验/2026-10-08_夜间方向实验/夜间结果报告.md`（不在本仓库）。
