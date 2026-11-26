---
name: identify-greybox-parameters
description: 用已知二阶方程和实验输入输出辨识系数，核验输入激励、局部可辨识性、自由递推误差与同名模型四参数或六参数接口。
---

使用 [二阶案例](../../../cases/matlab/README.md) 对应目录中的 MAT 数据，每行 [t,u,y]，初态零、输入零阶保持。确认系数顺序 [a,c,b,f0] 对应 y''=a y'+c y+b u+f0。先估时间间隔并检查是否均匀，再估计带界参数。避免拿顶层同名数据文件代替子目录的模型。

执行 `python -m controlkit.identify "../素材/55-系统辨识MPC-ADRC-PID" --out runs/identification`。交付四个带名字的拟合数值、历史参数、自由递推逐点轨迹和后段 RMSE。检查灵敏度矩阵秩 4，误差小于案例回归限 0.03。秩不足时改变激励或固定已知参数；边界拟合或偏置大时核实初值、常值扰动及采样。[辨识与估计](../../../knowledge/estimation.md)
