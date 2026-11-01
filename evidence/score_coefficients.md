# Score 系数核对（Saxena 2008 原文）

**日期**：2026-10-30
**来源**：原始论文 PDF，取自 NASA 官方 C-MAPSS 包（`data/nasa_cmapss.zip` → `CMAPSSData.zip` → `Damage Propagation Modeling.pdf`），已存 `docs/Damage Propagation Modeling.pdf`；文本 `docs/saxena2008.txt`。
**对应证据缺口**：`pre_draft_check_20261026.md` U1（a1/a2 待核）。

## 1 原文（第 VII 节，式 11 后）

> where s is the computed score, n is the number of UUTs, d = t^RUL − t_RUL (Estimated RUL − True RUL), **a1 = 10, and a2 = 13**.

式 11：

```
s = Σ_i  exp(-d_i / a1) - 1,  d_i < 0
         exp( d_i / a2) - 1,  d_i ≥ 0
```

## 2 矛盾与采用口径

- 原文同段又说：“**late predictions were more heavily penalized than early predictions**”（晚预测罚得更重）。
- 按原文的 `d = 估计 − 真实` 与 `a1=10, a2=13`，`d<0`（早预测）的分母更小、罚得更重，**与文字意图相反**。
- 社区标准实现（PHM08 挑战与绝大多数 C-MAPSS 论文）为：
  `d = 预测 − 真实`；`d<0` 用 `exp(-d/13)-1`，`d≥0` 用 `exp(d/10)-1`。这与“晚预测罚得更重”一致。
- **本文采用社区标准**：`a1=13`（早预测）、`a2=10`（晚预测），并在论文中注明“按 Saxena 2008 式 11 的意图；原文系数标注存在歧义”。

## 3 影响

- `code/metrics.py` 的实现（`a1=13, a2=10`, `d=pred-true`）**不需要改**；`score_coeffs` 字段从“待核”改为“已核（原文有歧义，采用意图一致的标准口径）”。
- 主表仍以 RMSE 为主、Score 为辅；Score 只报总值（sum over engines），并注明系数口径。
- 若审稿人坚持原文 `a1=10,a2=13`，可用补充材料给出两种口径的 Score；不改主结论（RMSE 为主）。

## 4 证据标签

`已核实`（原文 PDF + 文本）；矛盾点属于原文自身，不是二手引用错误。
