---
name: verify-mpc-prediction
description: 核验离散 MPC 的增量预测矩阵，定位未来输出与逐步递推首次不一致的步和通道。用于第二个或后续控制增量的索引、时序和直接通道检查。
---

从项目根目录运行 `python -m controlkit.prediction cases/prediction/input.json --out runs/prediction/pass.json`。验证用户矩阵时附加 `--matrix candidate.json`，其形状为 (N·ny,N·nu)，按时间优先排列。JSON 示例定义 A/B/C/D/E、x0、上一动作、N 行控制增量和同长扰动。约定 y[k+1]=C x[k+1]+D u[k]，传入离散矩阵，核对提供方是否使用相同采样时刻。保持原坐标；使用本次控制前的输出时先转换时序。

输出包括逐步预测、递推输出、最大误差和从 1 起算的首次失败步与通道。至少使第二步增量非零，并分别激励每个输入；一条脉冲通过只能覆盖被激励的列。失败时先查增量累加下三角块和 D 的时间对齐，再检查离散化。返回码 1 表示验证发现差异。[案例](../../../cases/prediction/README.md) · [预测与约束](../../../knowledge/mpc.md)
