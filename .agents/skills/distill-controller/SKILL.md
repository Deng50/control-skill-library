---
name: distill-controller
description: 负责控制算法轻量化。根据对象与专家控制器设计采集，或接收已有专家轨迹，完成模仿策略导出、同对象 PGAC 接续训练及原对象闭环对照。
---

对于热管理三执行器专家，先绑定原对象、参考及速度预览、实际动作和热流记忆，再逐项重算专家代价。判断是否部署新策略以原对象独立闭环为准，验证集需保留直接模仿权重作为候选。需要跨生态时先走[逐拍移植核验](../port-dynamic-plant/SKILL.md)。 [专业 Skill](../distill-thermal-expert/SKILL.md) · [完整决策过程](../../../knowledge/distillation-decisions.md) · [运行案例](../../../cases/heat-policy/README.md)

先核对部署可得观测、动作范围、历史长度、周期和参考可得性。对象和专家齐备时按[专家闭环采集](../collect-expert-trajectories/SKILL.md)规划工况。已有数据按[模仿策略](../clone-control-policy/SKILL.md)校验顺序、量程和整轨划分后直接训练。只需模仿时交付推理包和动作 RMSE。

需要优化时按[PGAC](../warmstart-pgac/SKILL.md)核验检查点、感知输入与成本梯度，冻结验证集选择，再在原对象评价。交付逐工况专家、模仿、两个候选对照，列出受益及退化指标。端到端执行 `python replay.py pgac`。[案例](../../../cases/spring/README.md) · [版本差别](../../../knowledge/pgac.md)
