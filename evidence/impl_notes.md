# 实现修正与主配置记录（2026-10-30）

本文件记录 E1/E2a 期间发现的实现问题、修正与主配置选择依据。所有数字可在 `results/` 找到；旧结果保留，不删除。

## 1 分块读出（E1 后修正）

- **问题**：初版分块实现（沿用探针）在一个块内所有 token 的读出都用**块边界记忆** `S_b`。L=30、C=16 时前 16 个 token 的读出几乎不随输入变化，等于浪费一半窗口。
- **修正**：块内逐 token 读出，用部分转移积 `G_{t-1}`：`y_t = S_b (G_{t-1} q_t)`，其中 `G_t = G_{t-1} A_t + B_t`。与逐 token 定义一致；chunk=1 时与探针逐 token 递推差 3.3e-9（`code/_test_models.py`）。
- **代价**：B=256/L=30/C=16/4 次扫描，实测 **~165 ms/iter**（旧实现 ~86 ms），峰值显存 756 MB（旧 475 MB）。
- **效果**（FD002 种子 0，同 4 epoch）：val_trunc 16.14 → 15.47；gate 16.42 → 16.86（gate 为独立 split，噪声大）。

## 2 α 饱和导致 NaN（E2a v0 发现）

- **现象**：FD002 种子 1 从第 6 epoch 起 loss=NaN（1772 步）；FD004 种子 0 也从第 3 epoch 起 NaN（2174 步）。早停回滚到 best（ep 4/2），但 v0 的本文模型结果不可用。
- **诊断**：输入有限、梯度范数 0.02–0.08、参数最大 ~0.4，非梯度爆炸。`alpha = sigmoid(·)` 在 float32 下可饱和到精确 1.0 → `eta_inj = (1-alpha)/r = 0` → `exp(-eta/eta_inj)` 前向为 0，反向为 `0×inf = NaN`。
- **修正**：`alpha = alpha.clamp(1e-6, 1-1e-6)`（`code/models.py`）。修正后 FD002 种子 1、FD004 种子 0 各 8 epoch 无 NaN；v1 门禁全程监控 `nan_steps`。
- **论文写法**：作为数值保护（numerical safeguard）写入实现细节，不改变稳定性命题的假设（α∈(0,1)）。

## 3 优化器补齐（对照 01 §6.8）

- 初版训练脚本漏了 **余弦退火** 与 **梯度裁剪 1.0**，已补：`CosineAnnealingLR(T_max=epochs)`、`clip_grad_norm_(1.0)`。
- 实测 TCN 早停在第 2 epoch，余弦尚未起作用；SRNL 在 lr=3e-4 时 best epoch 10，余弦开始起作用。
- 最终预算：**最多 40 epoch、patience 15、T_max=40**（不是 seeds.yaml 的 150 上限；150 未用，属预算上限）。

## 4 主配置对照（FD002，种子 0，只用验证集选）

| 配置 | val_trunc RMSE | gate_trunc RMSE | best epoch | ms/iter |
|---|---|---|---|---|
| C16, lr1e-3 | 15.468 | 16.862 | 4 | 157.5 |
| C16, lr3e-4 | **15.632** | 15.834 | 10 | 165.0 |
| C8, lr3e-4 | 15.854 | 15.152 | 10 | 152.5 |
| C16, lr3e-4, LayerNorm | 16.481 | 16.329 | 10 | 167.4 |
| B5 TCN, lr1e-3（对照） | 15.935 | 15.265 | 2 | 4.7 |

**选择规则**：只看 `val_trunc`（验证集），不看 gate；C16/lr3e-4 与 C8/lr3e-4 的 val 差 0.22（<1%），按“更接近构思的 C=16、结构更简单”选 **C16**。C8 留作后续 chunk 消融（E3），不在门禁里挑配置。

## 5 主配置（v1 起锁定）

| 项 | 值 |
|---|---|
| 输入 | 15 传感器 + 3 工况设定，L=30，cap=125 |
| 归一化 | FD001/FD003 全局 z-score；FD002/FD004 k-means(k=6)+簇内 z-score |
| 模型 | 双向 2 层 SRNL，dim=64（4 头×16），chunk=16，逐 token 读出，α 钳制 |
| 优化 | AdamW lr=3e-4（本文/B8）、1e-3（B1/B2/B5），wd=1e-4，batch=256，余弦 T_max=40，裁剪 1.0 |
| 训练 | ≤40 epoch，patience 15，监控 val 全窗口 RMSE |
| 门禁 | FD002/FD004，2 种子，B1/B2/B5/B8 + 本文 |
| 测试 | 官方 test 只在最终表用一次；RUL 主口径 cap 125，补充口径 raw |

## 6 v0 结果处置

- `results/e2a_*`（v0）保留；`results/e2a_summary_v0.json`、`results_gate_v0.csv` 记录按预注册聚合规则的判定（KILL_F5），但 **v0 的本文模型有 2/4 次 NaN，判定无效**，仅作审计。
- v1 结果用 `e2a_v1_*` 前缀，分析脚本 `code/analyze_gate.py --prefix e2a_v1`。
