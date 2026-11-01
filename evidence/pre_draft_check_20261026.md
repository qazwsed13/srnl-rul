# G2 起稿前证据预检（2026-10-26）

**用途**：写作与实验前的事实底稿。每条标状态；未核实项不得写进论文正文。
**日期修正**：系统日期为 2026-10-29；`02` 的 P1 原按 10/26 起算，实际只剩 3 天到 11/01。**11-15 门禁不变**，E0 数据审计与门禁对照冒烟优先，其余顺延。
**上游**：`01_论文构思_v0.1.md`、`02_写作思路与流程_v0.1.md`。

## 1 已核实（可直接用）

| # | 事实 | 来源 | 备注 |
|---|---|---|---|
| 1 | C-MAPSS 由 NASA C-MAPSS 仿真工具生成，含传感器噪声；训练到失效、测试截断并另给真实 RUL | data.nasa.gov 数据集页；awesome-industrial-datasets 说明页 | 仿真属性已核实 |
| 2 | 四子集规模：FD001 100/100、FD002 260/259、FD003 100/100、FD004 248/249（训练/测试发动机） | awesome-industrial-datasets 说明页；与既有认识一致 | 可直接进表 1 |
| 3 | FD001、FD003 为单工况；FD002、FD004 为六工况；FD003 的“2”指故障模式数，不是工况数 | 同上（FD001/FD003 one operating condition; FD002/FD004 six） | **修正了二手来源“FD003 两工况”的说法** |
| 4 | FD001/FD002 单故障模式（HPC 退化）；FD003/FD004 两种并发故障模式 | 同上 | 可直接用 |
| 5 | 每行 26 列：发动机号、循环数、3 个工况设定、21 个传感器 | 同上 | 与数据文件列数一致（待下载后复核） |
| 6 | 训练集行数量级：FD001≈2.1万、FD002≈5.4万、FD003≈2.5万、FD004≈6.1万 | ScienceDirect 2026 论文（Junaid 等）摘要 | 仅作量级；以 W1 实际行数为准 |
| 7 | RUL 标签常用分段线性截断，上限 125 | arXiv 2605.02507 原文片段（“we cap RUL values at a maximum of 125 cycles”） | 已核实 cap=125 |
| 8 | 指标为 RMSE 与 Score；Score 非对称（提前/滞后惩罚不同） | Saxena 2008 原文（NASA 官方包内 PDF，2026-10-30 取得）；原文写 a1=10、a2=13，但文字称“late 罚得更重”，系数与意图矛盾 | **采用社区标准口径 a1=13、a2=10**（d=预测−真实）；详见 `score_coefficients.md` |
| 9 | 《机床与液压》2025, 53(22): 54–60 刊出《基于Transformer编码器和手工特征的航空发动机剩余寿命预测》（陈栋、黄国勇） | 两处二手引用：综述页、他文参考文献 | **目标刊有直接 RUL 先例**；卷期页一致 |
| 10 | 《机床与液压》审稿周期 2 个月（官网投稿指南原文） | www.jcyyy.com.cn 投稿指南 | 前序已核 |
| 11 | Li X, Ding Q, Sun J Q. Remaining useful life estimation in prognostics using deep convolution neural networks. RESS, 2018 | ScienceDirect 文章页 | DCNN 基线题录已核实 |
| 12 | Yu 等，Test-Time Adaptation for Cross-Domain RUL Prediction: When Does It Work and What Should Practitioners Use，Eksploatacja i Niezawodność，2026 | ein.org.pl 文章页（作者 J YU，2026） | **年份按 2026 记**；卷期页待核 |
| 13 | Nested Learning（Behrouz 等）NeurIPS 2025，arXiv:2512.24695；Hope 是 Titans 的变体 | NeurIPS 海报页、arXiv 页、Google Research 博客 | 理论主干题录已核实 |
| 14 | VisionHOPE 为 2026-09-27 预印本（arXiv:2609.33325），官方代码 MIT | arXiv、GitHub | 预印本标注 |
| 15 | C-MAPSS 完整镜像：mapr-demos/predictive-maintenance 仓库 `notebooks/jupyter/Dataset/CMAPSSData`，含 train/test/RUL 四组共 14 个文件（12 个数据文件 + readme + 原始 PDF） | GitHub API 目录清单（文件大小逐一列出） | **下载源已找到，尚未下载** |

## 2 未核实（写作前必须补或删）

| # | 事项 | 现状 | 处理 |
|---|---|---|---|
| U1 | ~~Score 的精确系数（a1=13、a2=10）与公式~~ | **已核（2026-10-30）**：Saxena 2008 原文式 11 + 第 VII 节；原文系数标注与文字意图矛盾，采用意图一致的标准口径 a1=13、a2=10 | 见 `score_coefficients.md`；论文注明口径 |
| U2 | Yu 等 2026 的卷期页 | 只有文章页 | 投稿前补；正文可先写“2026, Eksploatacja i Niezawodność” |
| U3 | 《计算机工程与应用》是否有 RUL 先例 | 本轮未查到（非穷尽） | 若首投 CEA，投稿前知网复核；否则以机床与液压为首选 |
| U4 | Transformer / TCN / Mamba 中文 RUL 论文的卷期页 | 有题名与期刊，缺卷期 | 引用前补齐；补不到就换可核实文献 |
| U5 | C-MAPSS 使用许可 | 数据集说明页写“Information not available” | 论文中写“公开基准，按 NASA 数据页使用条款”；不主张商用 |
| U6 | 各目标刊“网络首发是否算见刊” | 学院口径未确认 | 用户向学院确认；决定排期策略 |
| U7 | 传感器子集与窗口长度（14 vs 21；30 vs 32） | 需按所选基线统一 | W1 数据审计后定；论文中写清 |

## 3 对构思与写作方案的修正

1. **选刊**：原决策树“结果强→CEA、结果平→机床与液压”应改为：**机床与液压为首选**（已有 2025 年航空发动机 RUL 直接先例，审稿 2 个月）；CEA 仅在结果强、且愿意用“层次”换“先例缺失”时考虑。振动与冲击保留为冲高。
2. **数据事实**：FD003 为单工况、双故障模式；避免在表 1 把故障模式数误写成工况数。
3. **Yu 等年份**：按 2026 记，不写 2027。
4. **DCNN 基线**：Li 等 2018，不再留“Zheng 2017”。
5. **下载源**：镜像已定位，E0 可直接取数；下载后必须核对行数、列数、RUL 文件长度，再进管线。

## 4 安全写作提示

- 不写“首次将嵌套学习用于 RUL”；写“据检索所及，尚未见……（非穷尽）”。
- 不写“保证预测精度”；只写“保证记忆范数不增”。
- 文献数字不进主表；主表只放自跑数字。
- 机床与液压的 RUL 先例只用于“期刊对口”，不用于“方法对比”（不同数据/协议）。
- VisionHOPE 一律标“预印本”；理论主干引 Nested Learning。
