---
name: clone-control-policy
description: 直接接收专家轨迹或闭环采集结果，训练小型模仿控制器，验证动作时间对齐与执行器量程，导出权重、训练集缩放和在线推理入口。
---

收集结果含配套 manifest 时运行 `python -m controlkit.policy imitate --data runs/expert/expert.npz --manifest runs/expert/data.json --out runs/imitation`。已有兼容 NPZ 可先运行 `python -m controlkit.expert_import expert.npz --split split.json --dt 0.2 --force 4 --initial 0 --out expert-meta.json`，然后替换数据与 manifest。档案数据缺 next_x 仍可模仿，进入 PGAC 前需真实配对转移。保留实际单位、字段顺序，见[案例](../../../cases/spring/README.md)。

观测 [位移,速度,参考,扰动]，监督标签为本步专家力。若部署看不到速度/扰动或专家含明显内部记忆，补估计器、因果延迟窗口或时序网络再验证；本例使用 4→48→32→1，tanh 映射 ±4 N。训练前逐轨检查 previous_u[k]=u[k−1]，先按整轨划分，再用训练集估缩放。只按验证损失选权重。交付 weights.pt、interface.json、heldout_action_rmse_N；执行 `python -m controlkit.policy infer runs/imitation 0.1 0 0.4 0.12` 返回 N。核对复载动作和量程，更改量程需重训或验证适配。[PGAC 交接](../../../knowledge/pgac.md)
