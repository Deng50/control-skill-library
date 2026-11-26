---
name: fit-conditional-thermal-pinn
description: 用现有温度轨迹或已知热平衡方程，训练恒定功率条件下的温度 PINN；与相同划分的纯 NN 对照并导出独立推理接口。
---

输入 CSV 的六列与工况拆分见[热过程案例](../../../cases/thermal/README.md)。字段按名解析。表中只有 I/O 时先确认 T0、P 的在线含义。物理路径必须给出 C、G、Ta、单位和恒功率声明；只有数据时省略 `--physics`。其他物理对象应先写新的残差算子并用已知解验证，不能改名套用热方程。

从库根执行 `python -m controlkit.thermal train --csv runs/thermal/data.csv --physics cases/thermal/physics.json --out runs/thermal/paired`。相同种子、网络、训练预算、验证集与留出条件对照物理开/关。输入 (N,3)，输出 (N,1)。用验证误差选择权重，报告逐工况测试 RMSE、初值误差和方程残差。归一化导数和输出用途见[物理约束](../../../knowledge/physics.md)。`python -m controlkit.thermal infer runs/thermal/paired/pinn 20 298 15` 返回 K。按需停止在数据分析或导出。高物理损失先核对单位和链式系数；普适精度门限由任务给定。
