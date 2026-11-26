# 原热管理离散模型的局部线性化

来源为素材/pgac/thermal-management-model.py 中 ThermalNominalModel.forward。用三个状态（电池温度 K、电机温度 K、SOC），三个动作（泵速、加热功率 W、阀开度），以及环境温度扰动。车速序列和 C0 固定在脚本声明值；只验证该工作点附近的一步映射。实际源文件指纹与点、扰动范围、偏置均在结果 JSON。

```powershell
python -m controlkit.linearize "../素材/pgac/thermal-management-model.py"
```

三十组邻域扰动逐通道与原模型比较，最大容许误差为 [0.002 K, 0.002 K, 1e-6 SOC]。额外用光滑解析函数及单侧斜率不一致的分段函数验证算法。跨查表边界、环境量或速度变化时重新选择工作点和验证包络。[矩阵、探针与实测误差](../../runs/linearization/report.json)
