---
name: fit-state-flow
description: 将二阶连续动力学和分段保持的变输入转换为可递推的 NN/PINN，支持原模型采数及已有同模式轨迹，导出物理坐标的下一状态预测函数。
---

1. 从原模型核实积分器顺序、初始状态、输入保持及采样间隔。对照四参数归档 [a,c,b,f₀]。输入只有位移时先确定上线如何得到速度；审查器不能把未来位移或内部真速度送给部署模型。[具体选型与故障诊断](../../../knowledge/model-decisions.md)
2. 从库根运行 `python cases/object-plant/collect.py`，读取 runs/object-plant/source_check.json 与 manifest.json。逐点对照原 MAT 后，按完整工况生成独立数据。用户已有匹配的数据时直接从训练入口读取，异构 CSV 先按 [时序数据审查](../audit-train-tabular-model/SKILL.md)检查字段、单位、分区与未来目标。核对状态和标签可得时间后再组窗。
3. 运行 `python cases/object-plant/train.py --data runs/object-plant --out runs/object-plant/paired --physics cases/object-plant/physics.json`。同一保持区间内给网络提供当前状态、动作和局部时间，比较同样本 NN 与带二阶方程残差版本。训练前按已知轨迹验残差单位与时间缩放。只有数据时省略 --physics，输出选另一个目录，明确当前基线及要补充的物理参数。
4. 运行 `python cases/object-plant/verify.py`，复载部署包，逐拍改变输入和保持时长，将自身输出作为下次初态，核对自由递推与局部方程残差。超出验证的采样间隔或幅值时先扩展工况。按[案例的目录及指标口径](../../../cases/object-plant/README.md)交付。
