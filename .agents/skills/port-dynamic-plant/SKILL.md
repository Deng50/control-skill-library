---
name: port-dynamic-plant
description: 把有积分器、延迟或惯性的 Simulink/Python 对象接入另一生态，提取复位与步进接口，以原模型实测状态核对单步和连续滚动并定位首次分歧。
---

先从用户的用途决定移植范围。专家采数可直接运行 SLX；需要 PyTorch 策略梯度时，把可微预测器与原对象闭环评分器作为两个入口。画出每个输入从命令到实际量的路径，标注持有值、上一执行量、积分器初值、延迟缓存及是否可观察。先明确 step 返回更新前还是更新后输出，然后实现 reset/step；有输出直接通道时单独对时。
   
从库根运行 `matlab -batch "addpath(fullfile(pwd,'cases','plant-transfer')); export_traces('../素材','runs/plant-transfer')"`，再运行 `python -m controlkit.parity`。具体对象、参数来自[原始二阶模块与端口](../../../cases/plant-transfer/README.md)。对其他对象，向 `controlkit.parity.compare(time,inputs,states,reset,step,tolerance,channels)` 传 K+1 个时刻/完整状态、K 行动作和逐通道物理容差。若只能测到输出，先比较相同复位后的完整输出轨迹，记录隐藏状态不可直接核验；不要按观测值强制复位内部状态。保持与原模型相同的输入序列、单位和初值。

返回每个模式的通道最大误差以及首个错误时刻、通道、原值和移植值。先查首个有输入变化的时间，接着查输入保持和历史寄存器，最后查积分器及精度。只有跨工况和边界通过后，才将副本用于后续 MPC 或训练。[源程序](../../../controlkit/parity.py)
