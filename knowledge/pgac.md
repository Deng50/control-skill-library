# 从专家到有限时域 PGAC

专家记录的是执行动作和当时可得观测，监督标签 u[k] 不得作为本步输入。若输入需要 u[k−1]，每轨首行按复位协议初始化；随后只用各控制器自己的上一实际动作。先做教师数据上的动作误差，再把每个策略放回原对象独立滚动。预测器的状态维度、学生的可见特征和动作维度分别定义。

热管理归档 pgac_auto_train_agent_v1/kernels/pgac_core.py 提供 Guidance、PerceptionNet、ActorNet 和 MPCCritic。默认感知 6→3，Actor 4→3，三个动作为泵速、加热功率及阀开度。原 MPCCritic 的参数中有 perception，但状态递推使用名义模型；Critic 是可微的有限时域代价计算。续训所用 actor_state/perception_state 和优化器状态区别于独立模仿权重。

本库的弹簧版本使用同一对象的学生 Actor。感知层对 [q,v,u,d] 拟合下一状态误差，线性对象可用无偏置最小二乘训练一层线性网络；要求训练数据列满秩。冻结该层，在耦合分支沿模型、感知层及 Actor 构建可微递推；保持输入梯度，更新 Actor 权重。J 包括归一化跟踪、力和力增量，预测越界用附加惩罚，执行时仍在原对象检查物理约束。无残差分支作为配对基线。若模型不充分、损失偏好变动或动作限幅改变，应重新验证，不能把弹簧策略直接导入热管理。

弹簧案例中的 Guidance 由每条工况的位移参考 r 与可得扰动力 d 提供，在六步滚动内保持。热管理内核用 ReferenceBank 与速度序列提供参考和外生输入。运行 `python -m controlkit.thermal_critic_probe` 可以直接观察原热内核的真实三步滚动。探针为 Perception 注册调用计数，再将所有感知参数加 3；其前后 J 值相同，感知调用数为 0，Actor 梯度仍然非零。热内核验证仅覆盖这段计算。[原内核探针](../runs/thermal-critic/report.json) · [弹簧两种滚动及闭环对照](../runs/pgac/report.json)
