# 原 Simulink 模块移植与逐拍核验

使用 55-系统辨识MPC-ADRC-PID/Linear_2nd_order_model4Ident.slx 的原始子系统。Sum → Integrator(v) → Integrator1(q)，位置和速度反馈系数分别为 c 和 a；参数取 [a,c,b,f₀]=[−4.2,−9,9,1]。临时副本在原速度信号上加日志，仅对工作区输入显式指定零阶保持。三条独立工况从 [0,0]、[0.4,−0.15]、[−0.6,0.3] 起步，施加双向阶跃、连续变化的采样激励和交替脉冲。0.01 s 固定步 ode3。源模型未声明位移的物理长度单位，因此容差沿用原坐标单位。

从项目根运行以下命令。

```powershell
matlab -batch "addpath(fullfile(pwd,'cases','plant-transfer')); export_traces('../素材','runs/plant-transfer')"
python -m controlkit.parity runs/plant-transfer
```

先用 reference[k] 的完整状态复位 Python，核对每个独立的下一状态。然后只在起点复位，沿全部动作自由运行。两个模式都要求各通道误差 ≤1e−10。在同一条轨迹上把动作延迟一拍，应输出首次分歧的步数、通道和参考值。将公用 compare 函数用于带隐藏状态的对象时，应导出完整记忆或提供重建历史；无法取得时只能从复位开始进行自由响应比较。通过本例后才将该接口用于生成神经网络训练数据或替代原对象评分。[原生轨迹元数据](../../runs/plant-transfer/native.json) · [逐通道报告](../../runs/plant-transfer/parity.json)
