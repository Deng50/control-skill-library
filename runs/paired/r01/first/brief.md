请将三个工程子任务作为可独立调用的 Python 函数，保存在工作目录 solution.py。评审器用其他工况调用；可使用 NumPy、SciPy、PyTorch。提交前可自测。所有试验只处理本任务目录的材料。若该目录有 library，允许使用其中的 Skill 与工程资料。请在 receipt.json 写出实际查阅的相对文档路径（键 used，字符串列表）和自测命令（键 tests，字符串列表）。各题互相独立。

1. 对象训练。金属块以环境温度为参考，已知 C [J/K]、G [W/K]，加热输入 p [kW]。经销商给的学生网络 net 把批量二维输入 [归一化时间 z, 条件 p] 映射为归一化温度 (N,1)，t_min = time_origin_min + z * time_scale_min，T_C = y_shift_C + y_scale_K * net(...), 环境为 Ta_C。实现 heat(net,zp,spec)，返回能对网络参数反传的 (N,1) 方程残差，单位 W，采用“蓄热 + 散热 − 输入”的正号约定。zp 为 torch 浮点张量。spec 键为 C,G,time_scale_min,time_origin_min,y_shift_C,y_scale_K,Ta_C。若 spec 缺少物理参数 C 或 G，返回 None 供外层选择纯数据训练。张量可来自批构造、默认不求导。

2. MPC。风门的离散动力学与端口时序见 actuator.py。实现 lift(spec,N)，返回 (F,P)，使未来 N 次传感器读数（依次每次 step 的返回值）等于 F@[x0,memory0,u_prev] + P@du。du 是本拍起的 N 个动作增量，矩阵以时间优先。spec 键为 a,b,c,d,g，均为实数，见原对象。要求任意正预测步数、任意允许初值、连续多步非零动作增量，保留直接通道。

3. 轻量化。见 teacher.py。一个行为克隆模型 actor 的输入顺序和输出转换由该文件定义，输入每行的物理状态为 [温度偏差,历史实际动作]。实现 policy_cost(actor,perception,x,reference,last,cfg,coupled)，返回标量 PyTorch 张量，供 PGAC 更新 Actor。x、reference、last 均为 (B,1)，分别是当前物理温差、固定参考和上一实际动作。完整滚动 cfg["N"] 步，累计每轨折扣代价并对 batch 平均。按专家时序、限幅、归一化、执行器滞后和物理量计算各阶段损失。perception 的 (B,2) 输入为按列拼接的物理温差、当前实际执行动作，输出 (B,1) 下一温差残差。在 coupled 打开时加入状态更新并保留动作梯度。cfg 键为 N,beta,a,b,limit,center,scale,q,r,s,discount，取值遵循 teacher.py。不要更新传入 Actor 的参数。
