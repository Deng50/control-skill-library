# 控制智能体技能库、知识库和案例库

从项目根打开新的 Codex 会话，可以直接描述工程任务，也可以用 `python -m controlkit.lookup "核验MPC预测矩阵"` 查看推荐 Skill、来源、代码入口和验收报告。三个角色各有一个薄入口，共用十个专业 Skill。按单项任务只运行所需步骤。每个案例可脱离 Harness 运行。[任务索引](index.json) · [验收总表](runs/acceptance.json) · [材料及版本](sources.json)

| 角色 | 可直接使用的请求 | 单独复跑 |
|---|---|---|
| [对象模型自动化训练](.agents/skills/train-object-model/SKILL.md) | 用给定热过程采数，用相同的留出工况比较 NN、PINN；如果只有 CSV 也交付基线。 | `python replay.py pinn` |
| [控制算法设计与参数调优](.agents/skills/design-and-tune-mpc/SKILL.md) | 给二阶对象设计有约束的 MPC，找出预测矩阵中第一个错误，并报告调参是否增加动作变化。 | `python replay.py mpc` |
| [控制算法轻量化](.agents/skills/distill-controller/SKILL.md) | 采集专家闭环轨迹、训练模仿策略，再用同一对象的权重暖启动 PGAC，比较原对象闭环。 | `python replay.py pgac` |

## 运行环境与命令

本机验证环境为 Windows、Python 3.10.22、NumPy 2.2.6、SciPy 1.15.3、PyTorch 2.7.1，训练用 CPU。Python 核心依赖在 [requirements.txt](requirements.txt)。本机可执行以下命令；其他环境用安装这些依赖的 Python 替换解释器。全量回放另需 MATLAB R2024b + Simulink、Node.js、本地素材和相邻 control-agent-harness。归档热 Critic 的原始内核还导入 matplotlib，本机版本 3.10.9。

```powershell
Set-Location D:\projects\control-skill-library
$py = 'D:\projects\control-agent-harness\.venv\Scripts\python.exe'
& $py replay.py all
& $py replay.py verify
```

`all` 从请求检索开始，依次执行预测矩阵、调参、模型采样与两条训练分支、专家采集与两类策略训练、独立反例、历史材料回放、MATLAB 和汇总验收。单独运行 `archive` 可复跑原对象线性化、原热管理 Critic 调用检查、二阶辨识、已有专家数据训练和 Harness 签名核验；`matlab` 只回放原始 SLX。各命令刷新对应的 runs 文件。`verify` 读取已有产物复核，数值门限仅针对冻结的案例。runs/checks 保存独立检查的输入和原始结果。

## 从请求到验收的首例

请求“给定未来控制增量，定位预测矩阵首次出错的步和通道”。路由到 [verify-mpc-prediction](.agents/skills/verify-mpc-prediction/SKILL.md)，输入 [离散模型与三步增量](cases/prediction/input.json)，由 [prediction.py](controlkit/prediction.py) 独立计算两列输出。正确矩阵最大误差 1.39e−17，阈值 1e−12。换成历史分块索引方式构造的反例，第 3 步通道 1 矩阵值 0.099627028，递推值 0.100347028，误差 −0.00072。另一组含直接通道和非零扰动的 MIMO 测试覆盖六个独立增量，在另一个注入错误中定位到第 3 步通道 2。[正确结果](runs/prediction/pass.json) · [反例](runs/prediction/legacy.json) · [独立核验](runs/prediction/independent.json)

## 本次实跑结果

| 任务 | 工况与读数 | 交付产物 |
|---|---|---|
| 热过程 PINN | 14 条轨迹按 9/2/3 划分。NN / PINN 测试温度 RMSE 为 0.13750 / 0.15528 K，最大初值误差 0.09219 / 0.02087 K。PINN 方程残差 0.76459 W，独立有限差分复核差 1.22e−5 W。换列顺序结果一致，两类跨集合泄漏被拒绝。 | [训练对照](runs/thermal/paired/report.json)、[模型接口](runs/thermal/paired/pinn/interface.json)、[缺方程分支](runs/thermal/data-only/report.json)、[复核](runs/thermal/independent.json) |
| 弹簧有约束 MPC | 六组权重，用独立对象与三套工况。测试位移 RMSE 从 0.10138 降到 0.08171 m，同时力总变化由 5.98 增至 14.54 N，力 RMS 从 0.934 增至 1.331 N。输入、位移和速度越界均为零。错误周期被拒绝，重离散后完成复验。 | [配置](runs/mpc/controller.json)、[调参报告](runs/mpc/report.json)、[动作与状态轨迹](runs/mpc/selected.csv) |
| 专家模仿到 PGAC | 12 条轨迹、432 个样本，按 8/2/2 整轨划分。模仿动作测试 RMSE 0.16766 N。两分支导入权重后的动作差均为 0 N。留出工况 07/02 的位移 RMSE：专家约 0.06828/0.04770 m，模仿 0.06986/0.04865 m，名义滚动 0.06968/0.04846 m，验证集选出的感知耦合版本 0.06998/0.04882 m。耦合版本满足测试约束，两条测试轨迹跟踪均较模仿略差。 | [专家数据](runs/expert/data.json)、[学生](runs/imitation/report.json)、[PGAC 对照与耗时](runs/pgac/report.json)、[部署接口](runs/pgac/selected/interface.json)、[反例与独立闭环](runs/pgac/independent.json) |
| 原生与专项复核 | 原 SLX 回放 2001 点，输出最大差 4.44e−16；四/六参数误用被拒绝。归档专家数据重训动作 RMSE 0.15692 N。原热管理 Critic 实际滚动三步，更改感知参数前后 J 同为 0.383109，感知网络被调用 0 次。 | [Simulink](runs/matlab/report.json)、[参数辨识](runs/identification/report.json)、[热对象线性化](runs/linearization/report.json)、[存档数据重训](runs/archive-import/imitation/report.json)、[热 Critic 探针](runs/thermal-critic/report.json) |

## 来源与适用范围

**本次实跑**包括以上核验及现有 Harness 的 9 个原生工具签名、27 个操作 schema 查询。文件级来源清单记录所用本地原件和摘要，案例 README 说明哪些实现为自编、哪些配置来自存档。[Harness 适配](knowledge/harness.md) · [版本与状态](cases/legacy/README.md)

**历史存档**包括既有热管理及弹簧策略权重、原 NMPC 与历史训练报告、电池 RLS/EKF/DEKF、PID/ADRC/LQR 参考实现。历史产物和本轮生成的 JSON 分别保留。本轮更改感知参数的热内核探针只验证 Critic 的实际连接，弹簧才完成了端到端模仿与 PGAC 训练。[案例状态](cases/legacy/README.md) · [辨识与估计知识](knowledge/estimation.md)

**仍待验证**包括客户现场数据与阈值、受限材料的外发授权、热管理全链路模仿权重导入与联合闭环、真实设备决策时延，以及新对象的模型递推与失效回退。现有 PINN 预测恒功率条件轨迹；任意时变输入下的滚动 MPC 应使用通过独立动态校验的转移模型。本机耗时分布按策略与工况写入报告，可在目标硬件复测。[物理建模](knowledge/physics.md) · [MPC](knowledge/mpc.md) · [PGAC](knowledge/pgac.md)
