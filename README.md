# Agent-ABot

基于 [AgentNav](https://github.com/JDR-neu/AgentNav) 的 ABot POI-Goal 导航适配。项目保留语义规划与 skill/tool 流程，并通过 `agentnav/abot` 中的 Harness 和执行器接入 ABot 仿真评测。

- `agentnav/abot/`：ABot Agent、规划、运动执行与配置。
- `agentnav/abot/workspace/skills/`：导航阶段的 skill。
- `scripts/`：评测及模型服务脚本。
- `tests/`：代码回归测试。
- [`docs/abot_poi_pipeline_cn.md`](docs/abot_poi_pipeline_cn.md)：从输入到输出的完整流程。

本机密钥与实验输出不纳入仓库。配置样例见 `agentnav/config/nanobot_config.example.json`；实际运行前应按环境填写本地配置，并参考项目文档准备 ABot 评测器、模型服务和数据。

源项目说明保留在 [AgentNav 中文 README](README_AgentNav_CN.md) 和 [English README](README_AgentNav.md)。
