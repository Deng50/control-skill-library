# 恒功率一阶热过程

根据旧 train-object-model 和 references/training.md 记载的条件热方程自编。physics.json 的 C=120 J/K、G=3 W/K、Ta=295 K 为本次演示取值。解析解生成各工况的参考温度；训练网络只读样本并通过自动微分计算残差。此参考对象是数学仿真。

从项目根运行（已有数据可跳过第一行）。
```powershell
python -m controlkit.thermal sample --csv runs/thermal/data.csv
python -m controlkit.thermal train --csv runs/thermal/data.csv --physics cases/thermal/physics.json --out runs/thermal/paired
python -m controlkit.thermal train --csv runs/thermal/data.csv --out runs/thermal/data-only
python -m controlkit.thermal infer runs/thermal/paired/pinn 20 298 15
```

CSV 列 trajectory、partition、time_s、initial_K、power_W、temperature_K。每轨从零时刻开始，T0 和 P 恒定；按轨迹分为训练/验证/测试，列顺序可换。参考案例 14 条轨迹，每轨 19 点，9/2/3 条划分。推理输入 t、T0、P，输出某个时刻的温度；恒功率查询不构成任意时变输入的转移模型。默认预算为 500 次 Adam 更新，按验证集选最佳轮。核验输出有限、工况分组互斥、量纲、物理分支有效、权重重新加载预测一致。双模型各保存权重、接口、损失曲线和测试逐点预测。在没有业务阈值时以实际误差交付。
