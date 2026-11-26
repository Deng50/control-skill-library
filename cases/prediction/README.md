# 二阶模型预测矩阵核验

本例根据课题组二阶对象 y''=−4.2y'−9y+9u+d 自编。Euler 采样 0.01 s。两行输出分别为 y 和 y'。原始 MPC_QP.m 在第 j>1 个增量的分块列只填了一个块。legacy-matrix.json 重现这一填充缺失，保留其余相同条件；这是针对该机制的反例。命令在项目根目录运行。

```powershell
python -m controlkit.prediction cases/prediction/input.json --out runs/prediction/pass.json
python -m controlkit.prediction cases/prediction/input.json --matrix cases/prediction/legacy-matrix.json --out runs/prediction/legacy.json
```

第一条须在 1e-12 内一致。第二条预期返回码 1，首次差异发生在第 3 步第 1 通道。外部坐标、扰动形状和采样定义见 input.json；候选矩阵以 JSON 二维数组输入。
