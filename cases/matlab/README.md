# 归档二阶 Simulink 与四参数辨识

本例使用素材/55-系统辨识MPC-ADRC-PID 中的 Linear_2nd_order_model4DataProduction.slx、Data4Ident.mat 和 model_param.mat，来源指纹见 [清单](../../sources.json)。顶层另有同名模型及六参数接口。只从选定案例目录取模型与参数，参数按 [a,c,b,f0]；六个值会在进入仿真前被拒绝。R2024b + Simulink 原生回放，复制模型到临时目录，以 SimulationInput 注入采数变量。在项目根执行：

```powershell
matlab -batch "addpath('cases/matlab'); replay_second_order('../素材','runs/matlab')"
python -m controlkit.identify "../素材/55-系统辨识MPC-ADRC-PID"
```

实际运行获得 2001 个 [时间 s, 输入, 输出] 样本，逐列与存档比对容差 1e-8。Python 独立采用精确 ZOH 和有界最小二乘，将单轨前 65% 用于拟合，后 35% 评估自由响应，灵敏度秩须为 4，留出 RMSE <0.03。该留出是同轨后段，跨工况辨识精度仍需另采独立激励。[仿真结果](../../runs/matlab/report.json) · [辨识结果](../../runs/identification/report.json)
