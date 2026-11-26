# 兼容旧 Control Harness

旧角色的三个 SKILL.md 和五份共享说明包含原生工具名称及分阶段 operation。用本机源代码核对，可从库根执行 `node adapters/inspect_harness.mjs`；可依次传插件 src、旧 skills、输出 JSON。结果记录注册文件、行号、源码哈希、参数声明原文，并实际调用本机 catalog 的 operationHelp 获取输入、配置、示例与允许角色。打开 JSON 按工具或 operation 精确检索。[签名快照](../runs/harness/signatures.json)

control_help 的 operation、agent 是可选查询条件。control_preflight 需要 agent、operation、workspace；inputs 与 config 按具体操作 schema 构造。control_prepare_review 依当前工具签名生成版本审阅。control_replay_plant_source 传 source 与 request 或 requestFile；control_build_zoh_plant 使用 Ac/Bc、initialCondition、sampleTime、inputNames，区分控制与扰动。control_configure_simulink_plant 配置现有块，control_wire_simulink 按真实端口连接，control_save_simulink_model 接受 model 名称，control_recheck_plant_workspace 使用空参数。实际下发任务仍取决于目标 Harness 会话和对应产物版本。本轮核对到源码与目录查询层，未通过 Harness 提交仿真作业；本库 Python/MATLAB 入口独立运行。
