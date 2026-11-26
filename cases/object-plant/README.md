# 从归档二阶对象训练可递推的 NN 与 PINN

**工程决定**　本例选择 PINN 作为该仿真对象的候选预测器。在四条独立测试轨迹上，从给定初态连续预测 8 s，其输出 RMSE 为 **0.00005282 Y**。同数据、同网络和同训练预算的监督 NN 得到 **0.0006001 Y**。两种模型都接收每一拍新施加的输入。原始材料只有仿真记录，输出和输入缺少完整的工程量纲，文中用 Y、U 表示原记录的数值单位。[逐工况与训练历史](../../runs/object-plant/paired/report.json) · [独立接口复核](../../runs/object-plant/independent.json)

## 1．先确认采到的信号和对象

取自素材子目录的采数脚本给出阻尼比 0.7、自然频率 3、增益 1 和偏置 1。SLX 中四项相加后依次经速度、位置积分器，记录实际施加的总输入与位置输出。得到 y″ = −4.2 y′ − 9 y + 9 u + 1。图上注释省略了偏置，实际加法器直接接入常量 f0。原参数辨识文件存放历史拟合值，本例生成新轨迹时取采数脚本的设定值。[原采数脚本](../../../素材/55-系统辨识MPC-ADRC-PID/Linear_2nd_order_Sys_Model4Datacollection.m) · [原 SLX](../../../素材/55-系统辨识MPC-ADRC-PID/Linear_2nd_order_model4DataProduction.slx) · [原 MAT](../../../素材/55-系统辨识MPC-ADRC-PID/Data4Ident.mat) · [物理参数及单位](physics.json)

| 量 | 时刻与用途 | 来源 |
|---|---|---|
| t，s | 每 0.01 s 的采样时刻 | MAT 第一列 |
| u，U | 区间左端实际施加值，零阶保持，正输入增加正向加速度 | MAT 第二列、SLX 输入记录端 |
| y，Y | 左端输出及训练标签 | MAT 第三列、位置积分器 |
| v，Y/s | 初次复位提供，后续由模型内部传播 | 原归档零初速，新仿真记录复位指令，训练窗口用过往 y 估计 |

库中采数 Skill 提供了原 SLX 的 MATLAB 原生回放。这里另以源文件 SHA-256、积分器连线、固定步长 ode3 配置为依据实现可批量复位的等价对象。先重新构造 5 s 周期、50% 占空比方波，从零状态回放 2001 点，时间、输入、输出对原 MAT 的最大绝对差为 3.55×10⁻¹⁵ s、0 U、4.00×10⁻¹⁵ Y。两组积分器状态在每条新轨迹开始前独立复位。对全部 19 条新轨迹再用连续线性方程的矩阵指数逐点对照，最大位置差 1.93×10⁻⁶ Y。训练包保留了原 MAT 的实存行。[等价对象与取数实现](collect.py) · [此次源文件与数值核验](../../runs/object-plant/source_check.json) · [库中原生回放](../matlab/README.md)

## 2．先分工况，再构造训练窗

原方波只能覆盖零初值和两个输入电平。另采独立初始位置、初速、随机种子、幅值、输入偏置下的多电平阶跃、双频正弦和随机两电平切换，测试另含扫频。每条新增工况 8 s、801 行。按整轨先分为训练 13 条（归档 1 条加新采 12 条）、验证 3 条、测试 4 条。编程检查时间间隔、数据指纹、同复位同输入的重复采样、区间内输入是否被保持。划分后从训练集取得 1160 个 0.1 s 窗口，每窗包含 10 个原生采样的输出标签。每条轨迹的首窗使用复位初速，后续窗口以左侧五点的单边差分估算起始速度。现场缺少速度传感器时，冷启动需要已知复位速度或经验证的速度估计器。[轨迹定义、分区、初态与种子](../../runs/object-plant/manifest.json) · [实采与等价采样数据](../../runs/object-plant/trajectories.npz)

训练特征逐维上下界为 y ∈ [−0.758, 1.162] Y、v ∈ [−2.708, 2.683] Y/s、u ∈ [−1.297, 1.280] U。将这些界存入接口供工况审查。本例训练和验证沿用理想的无噪声位置采样，五点微分遇到传感噪声时需重新评估。验证中有一条轨迹的最小输入达到 −1.306 U，测试中的所有输入均在训练输入范围内。[训练缩放](../../runs/object-plant/paired/pinn/scaler.json) · [接口与训练范围](../../runs/object-plant/paired/pinn/interface.json)

## 3．给时变输入选区间模型和物理损失

角色入口的热 PINN 示例假定整轨功率恒定。本对象可在每一控制拍改变 u，因此把网络输入设为当前状态 (y₀, v₀) 和本拍保持的 u。一个含 24 个 tanh 隐藏单元和线性旁路的网络给出三个加速度系数 A₀、A₁、A₂。设 h = 0.1 s、s = τ/h，区间解及其解析导数为

$$
\begin{aligned}
y(\tau)&=y_0+v_0\tau+\tau^2(A_0/2+sA_1/6+s^2A_2/12),\\
v(\tau)&=v_0+\tau(A_0+sA_1/2+s^2A_2/3),\\
y''(\tau)&=A_0+sA_1+s^2A_2 .
\end{aligned}
$$

这样在 τ = 0 严格满足位置和速度初值。监督 NN 的所有参数只由轨迹误差训练。以训练窗口的平均加速度 RMS 作为 Sₐ，各原生时刻的数据项为 (ŷ−y)/(½τ²Sₐ) 的均方，短时间段也能约束加速度。PINN 使用相同的数据项，另在训练特征范围内以固定 Sobol 种子生成 1024 个 (y₀,v₀,u,τ) 点，对 [y″−(a v+c y+b u+f₀)]/Sₐ 的均方赋权 1。残差各项使用 Y/s²，训练状态、幅值缩放和残差均按原有单位还原。配点只在区间内部求导，将上一段末端的连续 y、v 传给下一段即可更换输入。[训练与损失实现](train.py) · [源方程](physics.json)

两支使用 CPU float64、同一随机初始化、同一网络结构及同一 L-BFGS 预算，每阶段最多 25 次迭代，共 12 阶段。每阶段从三条验证轨迹各自的复位状态自由递推，按合并的位置 RMSE 选权。两支均选中第 12 阶段。省略物理配置后独立重训只输出 NN，报告列出缺失的方程和系数，独立复核确认其与成对对照中监督 NN 的划分、各权重和逐况指标一致。[成对运行](../../runs/object-plant/paired/report.json) · [无方程实跑](../../runs/object-plant/data-only/report.json)

## 4．逐工况验收并检查方程

每条轨迹只注入一次真实初态。随后连续施加 80 个 0.1 s 指令，使用模型自己预测的 y、v 递推。下表 RMSE 只计 80 个预测点。逐况的一步指标另使用实际位置和过去位置估算的初速，可据此判断误差是否沿长预测累积。[每条轨迹的一步、自由预测、峰值和末端误差](../../runs/object-plant/paired/report.json)

| 测试激励 | NN 一步 RMSE，Y | PINN 一步 RMSE，Y | NN 自由 RMSE，Y | PINN 自由 RMSE，Y |
|---|---:|---:|---:|---:|
| 随机多电平 | 0.00000902 | 0.00000442 | 0.0005544 | 0.00008685 |
| 双频正弦 | 0.00001130 | 0.00000358 | 0.0009628 | 0.00004506 |
| 随机两电平切换 | 0.00000914 | 0.00000219 | 0.0002283 | 0.00001969 |
| 新激励族扫频 | 0.00000801 | 0.00000334 | 0.0003923 | 0.00003462 |

在另外 96 个点分别用区间导数和中心差分计算微分方程残差。NN 和 PINN 残差 RMS 分别为 0.01773、0.002125 Y/s²。PINN 解析残差与独立差分的最大差为 1.10×10⁻⁷ Y/s²。同一算子对矩阵指数精确解的最大差分残差为 1.68×10⁻⁷ Y/s²。此处两种数值差均低于案例回归门限 3×10⁻⁶ Y/s²。[复核程序](verify.py) · [原始结果](../../runs/object-plant/independent.json)

## 5．将导出包接到时变输入端

部署包只依赖 NumPy，包含 [预测器](../../runs/object-plant/paired/pinn/predict.py)、[权重](../../runs/object-plant/paired/pinn/weights.npz)、[缩放](../../runs/object-plant/paired/pinn/scaler.json) 和[接口](../../runs/object-plant/paired/pinn/interface.json)。复制这四个文件到同一目录即可加载。加载时检查权重与缩放的摘要。调用 `step([y,v], u, dt)` 返回下一拍状态。`rollout([y0,v0], u_sequence, dt)` 返回含复位行的 (N+1,2) 数组。输入到达后保持 dt，支持 0.01 到 0.1 s。一个更长保持段可以连续施加相同指令。保留的每条模型轨迹依次记录 [t,u,y记录,y预测,v预测]。[PINN 预测轨迹](../../runs/object-plant/paired/pinn/rollouts.npz) · [NN 预测轨迹](../../runs/object-plant/paired/nn/rollouts.npz)

独立调用发布包，输入为 [−1,1] U 内逐拍更新的随机实数。以下三条 8 s 轨迹均从 [0.34,−0.67] 启动，与另写的平衡点加矩阵指数解比较。另从共同两拍历史分叉，把第 3 拍输入改为 +0.85 和 −0.85 U。前两拍状态相同，第 3 拍末 PINN 的位置差为 0.06635 Y。10 ms 下 PINN 的速度 RMSE 为 0.0001306 Y/s，略高于 NN 的 0.0001094 Y/s。选型依据是本任务的输出预测精度。[逐拍独立轨迹](../../runs/object-plant/independent_rollouts.npz) · [接口验证报告](../../runs/object-plant/independent.json)

| 指令保持时长 | 连续预测步数 | NN 的 y RMSE，Y | PINN 的 y RMSE，Y |
|---|---:|---:|---:|
| 10 ms | 800 | 0.0001152 | 0.0001003 |
| 50 ms | 160 | 0.00009542 | 0.00004701 |
| 100 ms | 80 | 0.0005636 | 0.00006505 |

当前结论限于该二阶仿真对象、已核验的输入和初态。真实设备仍需确定物理单位、冷启动速度误差、传感器噪声和控制器闭环表现。模型接口支持比较候选动作序列。本例未执行 MPC 闭环或硬件验收。

## 从库根复跑

另外运行 `python -m controlkit.native_flow_check`，可把同一部署包直接接到原 Simulink 输出的三条独立轨迹。重启每条轨迹后按记录的实际输入连续递推 100 步，分别核验位置、速度和逐步教师强制的误差。这一检查使用同级 plant-transfer 案例的 MATLAB 输出。[原生对象轨迹](../plant-transfer/README.md) · [神经模型实测](../../runs/object-plant/native_traces.json)

采数需要原素材目录。之后两条训练命令只读取已交付的轨迹包、清单和各自指定的配置。训练环境为 NumPy、SciPy、PyTorch，推理仅需 NumPy。四步依次生成数据、成对训练、缺物理训练和独立验收。

```powershell
Set-Location D:\projects\control-skill-library
$py = '..\control-agent-harness\.venv\Scripts\python.exe'
& $py -B cases/object-plant/collect.py
& $py -B cases/object-plant/train.py --physics cases/object-plant/physics.json --out runs/object-plant/paired
& $py -B cases/object-plant/train.py --out runs/object-plant/data-only
& $py -B cases/object-plant/verify.py
& $py -B runs/object-plant/paired/pinn/predict.py --state 0.34 -0.67 --u 0.2 -0.1 0.85 -0.85 --dt 0.1
```

最后一条命令给出四拍状态，末态约为 [0.209902,−0.558222]。迁移到其他对象时，先写明其实际采样定义、可获得的初态和可信方程。新的残差算子需通过已知解或独立数值解校核。此次试用补全了库中恒功率热轨迹之外的二阶变输入工程分支。[对象训练角色](../../.agents/skills/train-object-model/SKILL.md) · [采数技能](../../.agents/skills/sample-plant-trajectories/SKILL.md) · [恒功率热方程的原适用域](../../.agents/skills/fit-conditional-thermal-pinn/SKILL.md)
