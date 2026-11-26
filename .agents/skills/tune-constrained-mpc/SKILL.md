---
name: tune-constrained-mpc
description: 给定二阶对象、约束、目标或已有控制器，将跟踪、控制量及增量权重接入约束 MPC，搜索参数并用独立原对象闭环比较动作和约束。
---

核对 x(2)、参考位移、已知扰动力、上一实际力、Ts 和限幅。预测/控制步数满足 1≤M≤N；计算归一化跟踪、力、力变化惩罚并显式施加状态与输入限制。新对象先按[增量核验](../verify-mpc-prediction/SKILL.md)复核预测。[配置](../../../cases/spring/controller.json)包含冻结的调参、验证和测试场景，每组一个，不同组工况需互异。

从库根执行 `python -m controlkit.spring --config cases/spring/controller.json --dt 0.2 --out runs/mpc`。已有 Python 控制器传 `--controller-factory 模块:工厂`；工厂接收 (nominal_model,config)，实现 callable(x,r,d,previous)→u 和数值 dt，使用传入的 q,r,s。可用 `controlkit.spring:Controller` 验证此入口。连接不同对象时可在代码中用 rollout(plant,policy,scenario)；该对象需要 step、dt、force 和状态边界，当前闭环指标针对位移与速度。

读 report.json 的六组试验和复验，交付 controller.json、baseline.csv、selected.csv。先排除不可行与越界，再用固定外评分选权重；测试集只作最终评价。人为用 2Ts 对接应报错，重新离散预测器后复验。检查动作变化、RMS 力与 RMSE 取舍。求解器报错时记录工况并检查可达性和预测时域。[方法与代价](../../../knowledge/mpc.md) · [弹簧案例](../../../cases/spring/README.md)

若甲方给定权重范围，在配置的 parameter_space 中写 r/s 上下界，可另给 candidate_weights 的 [r,s] 二元组清单及 budget。基线占一个预算名额，超界候选跳过；超界基线报错。
