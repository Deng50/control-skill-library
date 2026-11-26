---
name: collect-expert-trajectories
description: 有原对象与已有专家控制器，需要决定采哪些闭环工况，记录执行动作、状态转移和轨迹划分，形成模仿与感知模型训练集。
---

列出传感器可得量和专家实际约束。初态取正负偏位及速度，参考跨方向且改变误差幅值，扰动取不同符号及幅值；用预跑确认能到达接近执行器边缘的样本。按对象允许范围制定工况，单独保留未见组合。参考 [expert.json](../../../cases/spring/expert.json) 及 [split.json](../../../cases/spring/split.json) 的十二条轨迹，复用/更改时记录工况来源。

先运行 `python -m controlkit.sampling_plan --pilot-steps 6 --out runs/expert/plan.json`。仅对训练工况预跑，按实际力距正负边界的距离分别给出保留工况或建议补试的初值、目标和扰动。本例正、负方向各有两个训练工况达到 3.92 N 的幅值阈值。[推荐工况与逐条预跑](../../../runs/expert/plan.json)

执行 `python -m controlkit.policy collect --controller runs/mpc/controller.json --scenarios cases/spring/expert.json --split cases/spring/split.json --out runs/expert`。验证采样周期、模型哈希、分区覆盖、轨迹第一条的上一实际动作、相邻样本状态连续。输出 NPZ 的 x/next_x(N,2)、u/previous_u/reference/d(N,1)、轨迹 ID 与时间(N,)，和 data.json 中靠近量程 98% 的样本数、工况与版本。零边缘样本时重新规划初始误差，记录补采范围。由[模仿训练](../clone-control-policy/SKILL.md)独立复查动作移位。[数据含义](../../../knowledge/signals.md) · [闭环案例](../../../cases/spring/README.md)
