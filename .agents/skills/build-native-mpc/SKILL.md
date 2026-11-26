---
name: build-native-mpc
description: 识别并改造现有 Simulink MPC 控制器，核对 MATLAB Function 或 S-function 实现、已连接的权重和上一实际动作，原对象闭环验证约束控制。
---

定位运行入口和源 SLX 的绝对路径。归档的根目录与 55 子目录各有一份同名 LMPC 模型；分别从 Constant 端口追踪实际权重。对于嵌入式 MATLAB 脚本先读 Stateflow chart 内容与外部连接，依据执行调用判断控制器类型。[原模型取证与重放](../../../cases/native-mpc/README.md)

1. 先回放素材顶层的 fmincon 闭环整定及 55 模型的基线，按原时间轴记录 r、y、u，改变一项权重确认实际输出发生变化。注意内部优化变量可能是 Δu，输出动作可能需积分，查实际接线。
2. 对离散模型区分扰动输入矩阵和输出矩阵。确定 Ts、N、动作决策变量，先用自由预测与独立状态递推核验，再构造 H/f、累加或差分映射、u 与 Δu 约束。把已执行的 u[k−1] 传给控制器。详细步骤见[建模到求解](../../../knowledge/controller-decisions.md)。复用 cases/native-mpc/native_mpc_prediction.m 和 native_mpc_step.m，入口参数与返回码见函数头。用独立 QP 对照可行解和活动边界。
3. 在副本中接入该控制器，运行 `run_native_mpc` 生成 SLX 及逐步 CSV。先冻结验证选择，独立负阶跃评价。交付时列出权重、求解状态、约束余量、跟踪/动作取舍，并区分普通 Simulink 仿真和目标代码实测。[闭环证据](../../../runs/native-mpc/report.json)
