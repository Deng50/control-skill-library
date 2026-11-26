交付一个可以替换现场基线的 MPC，用给定的 process.py 做联合仿真。它是本次新构造的阀门过程，输入为阀命令，原对象保留阀惯性，输出是带扰动的流量偏差。部署接口为 solution.plan(x,r,d,last,cfg)，x=[流量偏差,实际阀位]，r、d 为等长预览数组，last 为上一实际下发命令。返回字典 u（未来 N 个命令）、y（对应每次 step 返回的输出）和 J（专家内部代价），每拍仅执行首个命令。

请从原过程读出更新顺序并构建预测模型。内部代价按每步累计 q*(y-r)^2 + r_u*u^2 + s*(u-previous_command)^2。硬界为 cfg 的 lo,hi 和每秒 rate；采样秒数由 dt 给出。控制步数 M≤N，此后保持末个动作。工况提供初态、参考和可测扰动，假设未来预览可得。求解失败应有可观察状态。见 settings.json 的基线与候选权重及设计/验证工况。设计时按外部 position-RMSE + .015*RMS(u) + .03*RMS(Δu) 选型，在验证工况闭环对照现场基线。如果基线已经较好就如实报告。

从本目录运行 solution.py 应输出 summary.json（键 picked，是 settings.json 中候选的零起始编号，另给出设计及验证的实测读数）和 validation.csv（列 t,r,d,previous,u,y，使用实际执行的输入输出）。control callable 在未见的合法 cfg、初态与时域也应可单独调用。记录模型方程、求解与不可行处理；提交 receipt.json，used 列出实际读取的资料与 Skill，tests 列出实际执行的命令。
