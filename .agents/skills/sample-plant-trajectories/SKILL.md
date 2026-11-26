---
name: sample-plant-trajectories
description: 有对象模型，需要按工况激励、采集训练轨迹或在 MATLAB/Simulink 回放采数并与历史数据逐点校对时使用。
---

输入为可调用对象、复位初值、激励单位与范围、采样周期、结束条件和分组计划。先用小工况核对初始输出、激励符号、点数和时间轴，再扩大覆盖。相同条件的重复轨迹留在同一分区。[信号定义](../../../knowledge/signals.md)

- 热过程执行 `python -m controlkit.thermal sample --physics cases/thermal/physics.json --csv runs/thermal/data.csv`。用常值功率和多组初温，解析计算 14×19 个标签；首个点须与初温相等。将 CSV 传给[条件 PINN](../fit-conditional-thermal-pinn/SKILL.md)，附实际采样范围。[字段和工况](../../../cases/thermal/README.md)
- 二阶 SLX 执行 `matlab -batch "addpath('cases/matlab'); replay_second_order('../素材','runs/matlab')"`。核验时间、输入、输出与四参数归档独立对照到 1e-8；同名模型必须核对实际路径和参数长度。失败时定位首个有差异的信号并核对 solver、初值、参数和取样定义。[MATLAB 案例](../../../cases/matlab/README.md)

新对象先提供等价的 reset/step 或 SimulationInput 映射并定义终止判据；本地回放覆盖上述两个对象。
