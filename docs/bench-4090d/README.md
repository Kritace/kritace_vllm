# 4090D 实测材料（原始记录 + 溯源）

[07-实测汇总-4090D](../07-实测汇总-4090D.md) 的原始材料。全部来自 2026-09-27 的
**2 × RTX 4090 D 24 GB** 租用实例；服务器已关停，这里是全部可溯源记录。

| 路径 | 内容 |
|---|---|
| `raw/README.md` | 逐数字溯源索引：§六 往后每个数字 → 哪个日志的哪一行、用哪条命令复核 |
| `raw/logs/` | 39 个实验日志（`chain_*.log`、`evidence_*.log`、`bench_v6_spec.log` …） |
| `raw/scripts/` | 17 个实验脚本；`scripts/shell/` 里 29 个驱动脚本记录了精确命令与环境变量 |
| `raw/results/` | 14 个结果文件（各次运行的 token 序列 json、v4 的 `bench.jsonl`） |
| `raw/env/` | 7 个环境指纹（pip freeze ×2、sitecustomize 兼容层、机器指纹） |
| `raw/rental_out/`、`raw/v4_out/`、`raw/v5_out/`、`raw/probe.log`、`raw/watchdog.log` | 更早一批 v4 / v5 的原始日志 |
| `raw/pre_run2_root_logs.tar.gz` | 更早的 `root_logs`（4.2 MB 原始日志，压缩后入库） |

复现方式：脚本在 `raw/scripts/`，环境见 `raw/env/`。

两条注意：

- **不要直接套用绝对数字**：这批机器无 NVLink、P2P 被禁、all-reduce 走主机内存，换机器结论会变。
- **ZOS 的实测不在这里**：那是 RTX 3050 4 GB 上的数据，见 [../05-基准与复现.md](../05-基准与复现.md)。
