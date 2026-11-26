# 质量–弹簧–阻尼控制与轻量策略

归档来源为 素材/pgac/spring-damper-ppt-20261030/presentation/reproduce/case-config 下的 nominal、reference、controller、expert 四份 JSON。split.json 保留原专家数据工况划分。名义 m=1,c=0.8,k=2，独立参考对象 m=1.1,c=1,k=2.3。方程 m q''+c q'+k q=u+d，两状态 [q(m),v(m/s)]，输出位移，力和已知扰动力单位 N。示例以完整速度观测和可用扰动为前提。输入 ZOH，0.2 s。

本库自编有约束 MPC、有限组合搜索、MLP 和 PGAC；重新采样与重新训练，历史 GA 和历史权重另作参照。位移/速度边界为 ±1 m/±1.5 m/s，输入为 ±4 N；控制增量为软惩罚。预测和控制时域均 6 步。目标 r 在每个案例轨迹内恒定。验证前筛掉不可行候选；三套 MPC 调参、复验和测试工况来自不同初值、参考和扰动。

```powershell
python -m controlkit.spring --out runs/mpc
python -m controlkit.sampling_plan --pilot-steps 6 --out runs/expert/plan.json
python -m controlkit.policy collect --controller runs/mpc/controller.json --out runs/expert
python -m controlkit.policy imitate --data runs/expert/expert.npz --manifest runs/expert/data.json --out runs/imitation
python -m controlkit.policy optimize --epochs 20 --out runs/pgac
python -m controlkit.policy infer runs/pgac/selected 0.1 0.0 0.4 0.12
```

采集 12×36 步，8/2/2 整轨划分；NPZ 每行 x(2),reference(1),d(1),previous_u(1),u(1),next_x(2),t,trajectory_id。初始实际动作 0 N。策略特征依次为 q,v,r,d。上一动作供专家及 Critic 计算增量代价，学生网络未观测上一动作；若专家强依赖内部记忆，应加入因果历史并重新训练。训练标准化参数仅在训练集估计。小网络 4→48→32→1 输出 tanh 映射的控制力，加载时核验量程。

PGAC 两分支共用同一个模仿检查点。比较 6 步滚动代价，保持上一动作与扰动语义。所有策略从同一初态各自闭环，预热一次后记录本机单次调用均值/P95，不包含对象仿真。数值可复跑；计时受线程和机器负载影响。验收关注误差、约束、动作总变化和冻结选择。小样本结果只适用于所列工况。

已有专家数据可单独复跑。本例直接读取历史 expert_dataset-506f49fac84b 的 432 行，参照归档 artifact.json 确定周期 0.2 s 和力范围 ±4 N。该文件无逐行时间戳及 next_x，仅用于模仿训练。分区与表头校验后，留出动作误差 0.15692 N。[已有数据导入](../../runs/archive-import/data.json) · [重新训练报告](../../runs/archive-import/imitation/report.json)

```powershell
$expert = '../素材/pgac/spring-damper-ppt-20261030/artifacts/expert_dataset-506f49fac84b/expert.npz'
python -m controlkit.expert_import $expert --split cases/spring/split.json --dt 0.2 --force 4 --initial 0 --out runs/archive-import/data.json
python -m controlkit.policy imitate --data $expert --manifest runs/archive-import/data.json --out runs/archive-import/imitation
```
