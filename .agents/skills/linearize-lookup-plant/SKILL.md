---
name: linearize-lookup-plant
description: 将非线性、查表、带规则的对象在指定工作点局部线性化，输出预测矩阵、仿射偏置、单侧导数与原模型邻域误差。
---

输入一步映射 f(z)、工作点 z=[x,u,d] 与逐维扰动步长 h。函数 `controlkit.linearize.linearize(step,z,h,analytic=None)` 接受向量输入和向量输出；解析雅可比可用时与数值结果对照。连续导数和离散一步映射要分别声明；先检查相邻查表点、动作饱和与可行方向，再确定 h。

回放 `python -m controlkit.linearize "../素材/pgac/thermal-management-model.py"`。返回 runs/linearization/report.json，含 A(3×3)、B(3×3)、E(3×1)、C/D、偏置、半步变化与逐通道原模型误差。对左右导数差大的切换点，用单侧或分段模型，并在相同边界测试。超出温度或 SOC 误差预算时缩小适用域或保留非线性滚动。[工况与容差](../../../cases/linearization/README.md) · [离散预测](../../../knowledge/mpc.md)
