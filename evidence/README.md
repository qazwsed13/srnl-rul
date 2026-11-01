# G2 证据包（evidence）— 使用说明

**状态**：E0/E1/E6 已完成（2026-10-30）；E2a 门禁运行中。本目录文件均可在 `code/` 找到生成脚本。
**定位**：论文与投稿材料中**每一个数字的唯一来源**。写作阶段提示词会要求先读本文件。

## 硬规则

1. 论文里的数字只能取自本目录的 csv/json/txt；禁止手抄、估算、凭记忆填写。
2. 文献数字不进入主表；只允许在讨论中作“量级参照”，并注明“不同预处理，不可直接比较”。
3. 负结果、NaN 行、失败配置必须保留；不得因结果不好而删行、删种子或换指标。
4. 每个文件附一行生成命令与 git commit；环境版本写入 `env.txt`。
5. 逐单元预测必须落盘，统计脚本从预测文件复算，不直接读汇总表。

## 预期文件（W1 起）

| 文件 | 内容 | 建立时机 |
|---|---|---|
| `data_audit.md` | C-MAPSS 四子集事实、26 列、RUL 长度、NaN=0 | 已完成 2026-10-30 |
| `env.txt` | python / torch / cuda / GPU / 驱动版本 | 已完成 2026-10-30 |
| `seeds.yaml` | 5 个种子与完整配置（层数、宽度、C、α_init、lr、batch） | 已完成 |
| `sensor_subset.md` | 统一传感器子集（FD001 方案 15+3 维）与方差并集敏感性 | 已完成 2026-10-30 |
| `e1_smoke.md` | FD001 管线冒烟：5 模型、5 epoch、ms/iter、显存、无 NaN | 已完成 2026-10-30 |
| `e6_stability.md` | 范数比断言、γ=5 有/无控制、‖A‖₂/‖B‖₂ 抽样 | 已完成 2026-10-30（待按主配置重跑） |
| `impl_notes.md` | 分块读出、α 钳制、余弦/裁剪、主配置对照与锁定 | 2026-10-30 |
| `score_coefficients.md` | Saxena 2008 Score 系数原文与采用口径 | 2026-10-30 |
| `e2a_gate.md` | FD002/FD004 门禁：2 种子 × 5 模型、聚合与配对 CI、KILL 判定与救援决定 | 已完成 2026-10-30 |
| `results_main.md` | E2b 官方 test 主表（4 子集 × 5 种子 × 7 模型）、配对统计、故事 B 判定 | 已完成 2026-10-31 |
| `metrics_diagnosis.md` | test/gate 标签分布、尾部主导、晚段偏差、校准检查 | 已完成 2026-10-30 |
| `mechanism.md` | α/η 随退化阶段变化（5 种子 × FD002/FD004） | 已完成 2026-10-31 |
| `results_ablation.md` / `table3_ablation.csv` | E3 消融（FD002 3 种子 + FD004 2 种子） | 已完成 2026-10-31 |
| `results_robust.md` | E5a 噪声鲁棒性（gate，5 种子） | 已完成 2026-10-31 |
| `results_main.csv` | 四子集 × 方法（8 对照 + 本文）× 种子的 RMSE / Score / 耗时 / 显存 | W3–W4 |
| `results_ablation.csv` | 消融组 A/B/C，含 NaN 行与激活率 | W4 |
| `mechanism.csv` | η̃_t、α_t、软上限绑定率、谱钳制激活率（按工况/退化阶段汇总） | W5 |
| `stats.json` | 配对 Wilcoxon、Holm、rank-biserial、bootstrap 95% CI | W5 |
| `pred_<subset>_<model>_<seed>.csv` | 逐测试发动机预测值，供复算 | W3 起 |
| `figdata/` | 图 1–5 的绘图数据快照 | W5 |

## 与写作阶段提示词的关系

`00_论文构思工作单_v0.1.md` §7 的下一阶段提示词要求“先读 `evidence/README.md`”。若本目录尚未建立，先按本说明建立骨架，再开始写作；**不得在证据包为空时凭记忆写数字**。
