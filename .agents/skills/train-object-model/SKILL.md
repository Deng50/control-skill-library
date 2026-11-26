---
name: train-object-model
description: 负责对象模型自动化训练。按已有可运行对象或已有数据选择采数、数据审查、灰箱辨识、条件热 PINN 与纯 NN 训练，交付可加载模型及独立工况误差。
---

先从[信号约定](../../../knowledge/signals.md)写出输入、输出、单位、实际可测量、时间语义和工况标识。用户只委托一步时，直接选择对应专业 Skill。

1. 给了模型，按[运行对象并采数](../sample-plant-trajectories/SKILL.md)生成轨迹。给了 CSV，用[时序数据审查与训练](../audit-train-tabular-model/SKILL.md)检查映射与泄漏。
2. 若请求辨识方程参数，执行[灰箱辨识](../identify-greybox-parameters/SKILL.md)。若要恒功率热 PINN，检查方程来源、单位和初值后进入[条件 PINN](../fit-conditional-thermal-pinn/SKILL.md)，使用同一划分比较普通 NN。缺少物理信息则交付数据基线与缺项。
3. 交付预测入口、缩放、权重、测试工况误差及适用输入序列。若要进一步用于 MPC，另验证可接收任意允许控制输入的递推接口。回放 `python replay.py pinn`。[案例](../../../cases/thermal/README.md)
