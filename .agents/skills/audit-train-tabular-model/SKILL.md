---
name: audit-train-tabular-model
description: 用户已有多轨迹 CSV，要求映射列与单位、检验时序窗口、构建一步预测或缺乏物理方程时训练可独立推理的监督基线。
---

输入一个有表头的 CSV 和参照 [table.json](../../../cases/thermal/table.json) 的 schema。声明轨迹、分区、秒时间轴、dt、特征单位和 lag、目标单位和 ahead。每行窗口只在同轨取样。检查采样间隔、重名列、未来信息、同轨跨分区；相同工况换 ID 的复制轨迹还需按工况去重。

```powershell
python -m controlkit.table audit runs/thermal/data.csv cases/thermal/table.json --out runs/table/audit.json
python -m controlkit.table train runs/thermal/data.csv cases/thermal/table.json --out runs/table
python -m controlkit.table infer runs/table 296 10 295
```

本例三个特征为 T[k]、P[k]、T[k−1]，目标 T[k+1]。交付各分区有效行数、每目标原单位 RMSE、weights.pt、interface.json 和 test.csv。缺单位先补字段；时间不规则时应分段或先定义重采样；训练集常量列要检查激励。该一步基线要在多步闭环使用前另验。[信号与时序](../../../knowledge/signals.md)
