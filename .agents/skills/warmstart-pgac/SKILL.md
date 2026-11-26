---
name: warmstart-pgac
description: 以同一对象已训练的模仿 Actor 初始化有限时域 PGAC，验证动作一致性、感知残差对滚动代价的贡献和原对象闭环表现。
---

准备训练数据、meta、已选控制器和模仿策略（相对库根默认 runs/expert、runs/mpc、runs/imitation）。输入声明须同一对象、同一特征顺序、动作单位和量程。要求相对名义模型的真实下一状态标签，各轨迹第一条以前一动作初值开始，后续上一实际动作等于前条已执行动作。

执行 `python -m controlkit.policy optimize --actor runs/imitation --controller runs/mpc/controller.json --epochs 20 --out runs/pgac`。按[弹簧案例](../../../cases/spring/README.md)输出 nominal_only、residual_coupled 两个自编版本与 selected。原案例热管理滚动器的行为及梯度路径见[PGAC 方法](../../../knowledge/pgac.md)。训练仅改变 Actor。核验第一次更新前动作差为 0，同一 Actor 两种滚动 J 不同，训练集感知拟合在留出验证集有效，Actor 梯度非零。按验证集先冻结策略，再逐工况查看专家/基线/候选测试跟踪、幅值越界和耗时。发现更改量程先重映射并重新采数或验证；不得以加载同宽权重取代语义检查。
