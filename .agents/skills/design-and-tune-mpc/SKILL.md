---
name: design-and-tune-mpc
description: 负责控制算法设计与参数调优。针对对象和控制目标生成 MPC，或接入已有控制器，按独立工况完成预测核验、权重搜索与原对象闭环评价。
---

收到 SLX 先查实际编译的连续/离散周期、控制器 chart 与参数接线，随后回放原闭环，再将新控制器接入同一对象。需要转 Python 时先走[逐拍移植核验](../port-dynamic-plant/SKILL.md)。 [专业 Skill](../build-native-mpc/SKILL.md) · [完整决策过程](../../../knowledge/controller-decisions.md) · [运行案例](../../../cases/native-mpc/README.md)

从用户材料提取状态、可测输出、控制、扰动、参考、采样与限幅。[信号约定](../../../knowledge/signals.md)给出记录方式。明确参考和扰动的未来值来自何处、成本量纲、跟踪与动作权衡、软硬边界及不可行处理。

已有离散矩阵先用[增量预测核验](../verify-mpc-prediction/SKILL.md)。非线性或查表模型先做[工作点线性化](../linearize-lookup-plant/SKILL.md)。随后用[约束 MPC 调优](../tune-constrained-mpc/SKILL.md)生成基线或传入已有控制器工厂。接受一个子任务时直接返回其数值证据。端到端执行 `python replay.py mpc`，交付控制器、测试轨迹、动作和约束对照。[案例](../../../cases/spring/README.md)
