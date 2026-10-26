# run2 归档（2026-09-27）—— 数字溯源索引

这个目录把 `rental/实测结果汇总.md` 里**第六节往后的每一个数字**都对应回原始记录。
目的只有一个：**别人（包括三个月后的自己）能一条命令验证任意一个数字。**

```
run2_20260927/
├── README.md      ← 本文件（溯源索引）
├── logs/          ← 39 个日志（chain_*.log / evidence_*.log / bench_v6_spec.log / probe.log …）
├── scripts/       ← 17 个实验与诊断脚本（.py，可重跑）
│   └── shell/     ← 29 个驱动脚本（chain_*.sh 等，记录了精确命令与环境变量）
├── results/       ← 14 个结果文件（各次运行的 token 序列 json + v4 的 bench.jsonl）
└── env/           ← 8 个环境指纹（pip freeze ×2、sitecustomize.py 兼容层、机器指纹）
```

共 108 个文件 / 980 KB。**服务器已关停，这是全部可溯源材料。**

更早的一批（v4 / v5 的原始日志与 `probe.log`）在上一级 `fetched/` 里（`rental_out/`、`v4_out/`、`root_logs/`）。

---

## 一、§六 8B 表 —— `logs/chain_f.log`

```bash
grep "^8b-" logs/chain_f.log
```

| 文档里的行 | 日志中的原始行 |
|---|---|
| 单卡 eager 37.05 / 36.0 / 245.4 | `8b-single-eager \| ttft=37.05ms \| decode=36.0 \| batch8=245.4` |
| 单卡 cg 37.34 / 58.1 / 398.2 | `8b-single-cg \| ttft=37.34ms \| decode=58.1 \| batch8=398.2` |
| TP=2 eager 43.14 / 27.8 / 195.1 | `8b-tp2-eager \| ttft=43.14ms \| decode=27.8 \| batch8=195.1` |
| TP=2 cg 49.44 / 94.6 / 556.8 | `8b-tp2-cg \| ttft=49.44ms \| decode=94.6 \| batch8=556.8` |

> 引擎是**本仓库 v4**（`bench/bench_v4.py`）。这一节的两条"结论反转"（TP 变划算、CG 收益变小）
> 都能直接从这四行数字看出来，不需要额外假设。

## 二、§七 / §八 官方 vLLM 对照与量化 —— `logs/chain_p.log`

```bash
grep -E "^8b-vllm-(bf16|awq)-cg" logs/chain_p.log
```

| 文档里的行 | 日志中的原始行 |
|---|---|
| 官方 bf16：27.31 / 59.3 / 407.5 | `8b-vllm-bf16-cg \| ttft=27.31ms \| decode=59.3 \| batch8=407.5` |
| 官方 AWQ：26.04 / 156.0 / 930.7 | `8b-vllm-awq-cg \| ttft=26.04ms \| decode=156.0 \| batch8=930.7` |

> ⚠️ 写索引时亲自跑过一遍：直接写 `grep "^8b-vllm-"` 会**多出一条 `8b-vllm-spec-dummy`**，
> 容易和 §九 的投机数字混在一起，所以这里必须把模式收紧到 `(bf16|awq)-cg`。

⚠️ **同一个日志里 `8b-vllm-spec-real` 是 FAIL 的**（KV cache 不够），
**成功那次在 `chain_s.log`** —— 引用 §九 的投机数字时不要拿错文件：

```bash
grep "spec-real" logs/chain_p.log   # FAIL（错误的那个）
grep "spec-real" logs/chain_s.log   # 成功：ttft=73.27ms | decode=128.1 | batch8=580.2
```

## 三、§九 9.1 官方投机 —— `logs/chain_s.log`（real）+ `chain_p.log`/`chain_r.log`（dummy）

```bash
grep "spec-" logs/chain_s.log logs/chain_p.log logs/chain_r.log
```

- 真实文本：`8b-vllm-spec-real | ttft=73.27ms | decode=128.1 | batch8=580.2`（`chain_s.log`）
- dummy 两次（**两次数字不同，文档里如实写了两个值**）：
  `127.4 / 245.1`（`chain_p.log`）与 `119.6 / 233.8`（`chain_r.log`）
- 口径：`/root/venvs/vllm-official`，vllm **0.30.0**，target=Qwen3-8B，draft=Qwen3-0.6B，K=5，
  `--prompt dummy|real`，CG 开启

## 四、§九 9.2 自家引擎 K 扫描 —— `logs/bench_v6_spec.log` + `results/v6_spec.json`

```bash
cat logs/bench_v6_spec.log          # 打印成表的那份
python -c "import json;print(json.load(open('results/v6_spec.json'))[0])"
```

- 脚本：`scripts/bench_v6_spec.py`（三类提示词 × K∈{0,3,5,8}，`n=64`，3 次重复取均值）
- 口径：**本仓库 v6**，Qwen3-0.6B，单卡，`enforce_eager=True`，n-gram 草稿
- 同一个脚本还顺带做了**贪心一致性比对**（`greedy_match` 字段）—— 自然散文那几行的
  `★不一致★` 就是 §九 9.4 要解释的现象

## 五、§九 9.3 n-gram 接受率表 —— `logs/chain_final.log`

```bash
sed -n '/n-gram 接受率/,/^DONE/p' logs/chain_final.log
```

- 脚本：`scripts/spec_acceptance_probe.py`（纯 CPU 统计，单请求 256 token）
- `α` = 单步接受率，`E[token]` = 每步期望产出，`命中步 / 无草稿` = 统计样本分布

> ⚠️ 该脚本在更早的 `chain_i.log` 里曾因 `α=1.0` 触发 `ZeroDivisionError` 崩溃，
> 修好后在 `chain_final.log` 才拿到完整表 —— **引用时以 `chain_final.log` 为准**。

## 六、§九 9.4 「完全并列」的证据 —— `logs/evidence_01..05-*.log`

这 5 份日志是**专门为重跑判定实验而生成的**（脚本见 `scripts/`），每一份对应一个被排除的假设：

| 日志 | 脚本 | 排除了什么 |
|---|---|---|
| `evidence_01-numerical-tie.log` | `compare_step.py` | **决定性证据**：两种情形在同一步的元数据与 top-3 logits 对拍 → `13@+20.250 / 11@+20.250` 并列 |
| `evidence_02-canonical-ref.log` | `decide_v6.py` | 用「一次性 prefill」做规范参考 → 参考与 decode 核都回 `13`，只有投机回 `11` |
| `evidence_03-clean-history.log` | `isolate_v6.py` | 干净历史下重放同一上下文 → **四行 seg 全对**（问题出在历史） |
| `evidence_04-poison-kvpool.log` | `poison_v6.py` | **毒化整个 KV pool（填 100.0）→ 输出完全不变** → 核不读写指针之外的槽 |
| `evidence_05-no-corrupt.log` | `trigger_v6.py` | 喂真值草稿 + 注入单次部分接受 → **0/12 步不匹配** → 账本正确 |

重跑方式（**需重新租卡**）：

```bash
# 脚本已归档为 scripts/shell/chain_evidence.sh，连同 scripts/ 下的 .py 一起上传后：
bash chain_evidence.sh          # 5 组实验，约 8 分钟
```

⚠️ 这些脚本里的 `MODEL` / `PROMPT` 都是**硬编码**的，重跑前先确认模型路径还在
（Qwen3-0.6B 在 `/root/huggingface/Qwen3-0.6B`，可用 `env/dl_8b.py` 重新下载）。
⚠️ **服务器已关停** —— 这批脚本只能在新实例上重跑，环境按 `env/pip-freeze-*.txt` 重建。

## 七、§十 DP 与并发 —— `logs/chain_all.log` + `logs/mem_dp2.log`

```bash
# dp1 基准（同口径单卡对照）
grep -E "^0\.6b-dp1" logs/chain_after.log
# dp2 实测
grep -E "^0\.6b-dp2" logs/chain_all.log
# 并发扫描：表头在 chain_all.log，值紧跟其后
sed -n '/n=1 \/ tp2 cg/,/n=128/p' logs/chain_all.log
# DP=2 时两张卡的逐秒显存采样
cat logs/mem_dp2.log
```

| 文档里的行 | 日志中的原始行 |
|---|---|
| dp1 eager 710.0 | `0.6b-dp1-eager \| dp=1 \| reqs=32 \| 4096 tok \| 710.0 tok/s`（`chain_after.log`）|
| dp2 eager 1331.4 | `0.6b-dp2-eager \| dp=2 \| reqs=32 \| 4096 tok \| 1331.4 tok/s`（`chain_all.log`）|
| dp1 cg 4784.3 / dp2 cg 8260.2 | 同上，后缀 `-cg` |

> ⚠️ **`dp1` 与 `dp2` 不在同一个日志里**（前者在 `chain_after.log`，后者在 `chain_all.log`）
> —— 这是索引写完后逐条验证时才发现的，只看一个文件会找不到基准值。

> ⚠️ 并发扫描的日志格式是「表头行 + 紧跟的 `ttft_ms=… decode_tok/s=…`」，
> 所以能看到 `n=1` 那行是 `decode=0.00`：**`n=1` 时请求只生成 1 个 token，
> decode 吞吐这个指标本来就不成立**（文档里就写了“—（只生成 1 token）”，没有编造数字）。

## 八、§十一 缺口里提到的失败记录

| 缺口 | 证据文件 |
|---|---|
| 官方 vLLM 在 0.6B 上起不来（缺依赖 / `shm_broadcast cancelled`） | `logs/chain_d.log` |
| `PP=2 + cg + compile` 的 `RuntimeError: could not find 0x...` | 见 `fetched/rental_out/run_v4_*.log` |
| `chain_bc.log` 为空 | 该脚本当时没产出（`chain_all.log` 里有对应的补跑结果） |

---

## 九、环境指纹（`env/`）—— 重建环境时看这里

| 文件 | 内容 | 为什么留着 |
|---|---|---|
| `pip-freeze-miniconda.txt` | 197 行，跑 v4~v7 的 Python 环境 | 含 `torch @ …torch-2.8.0%2Bcu128…`、`flashinfer-python==0.7.0`、`transformers==5.17.0` |
| `pip-freeze-vllm-official.txt` | 198 行，官方 vLLM 对照环境 | 含 `vllm==0.30.0`、`flashinfer-python==0.6.18.post1` |
| `sitecustomize.py` | 15 行兼容层 | 踩坑文档 A9 引用的就是它；丢了要重写 |
| `machine-fingerprint.txt` | `nvidia-smi` + OS + CPU/内存 | 引用数字时要交代的硬件指纹 |
| `dl_8b.py` / `dl_moe.py` | 模型下载脚本 | 重建环境时用 |

**一条很实用的记录**：`pip-freeze-miniconda.txt` 里 flash-attn 是以
`flash_attn @ file:///root/wheels/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
形式记录的 —— 这串文件名就是**预编译 wheel 的五要素匹配**（cu12 / torch2.8 / cxx11abi / cp312 / linux_x86_64），
正好对应踩坑文档 A4。下次租卡装 flash-attn 直接照这个文件名去匹配，不用再试错。

## 十、已知的归档缺口（如实说明）

1. **`chain_g.log`（370 KB）是 MoE 模型的下载/删除日志**，与性能数字无关，留着仅为完整性。
2. **`dl_8b.log` / `dl_moe.log` 没归档**（共 4.5 MB，纯下载进度条）。
3. **模型权重没归档**（也没法归档）：
   - Qwen1.5-MoE-A2.7B（27 GB）在实验结束后删掉以腾数据盘
     → **§四（v5 MoE）的数据无法重跑**，但原始日志在 `fetched/rental_out/`
   - Qwen3-0.6B（1.5 GB）与 Qwen3-8B（16 GB）可用 `dl_8b.py` 重新下载
4. **本仓库 v7 的量化吞吐给不出** —— `QuantLinear.forward_fused()` 未接入融合 GEMM
   （刻意显式失败，不静默回退），所以 §八 的 2.63× 全部来自官方 vLLM。

> ~~没有归档 pip freeze~~ —— **已补上**（见第九节），引用依赖版本时不必再靠记忆。

## 十一、如果以后重新租卡：优先补这两项

现有数据的两个真实短板，**只有重新租卡才能补**（按价值排序）：

### 1. 本仓库 vs 官方的投机**同口径**对照（价值最高）

现状不可比：官方是 **8B**（decode 2.16×），本仓库是 **0.6B**（按文本类型 5.87× ↔ 1.16×）。
补法二选一：

- **省事**：在官方 vLLM 上跑 0.6B + 0.6B draft 的 K 扫描，与本仓库 §九 9.2 逐 K 对齐
  （同模型、同 K、同文本 → 直接可比）
- **更硬**：把本仓库 v6 的 `bench/vs_vllm.py --spec-k K` 跑到 8B（需 `--max-model-len 2048`、
  `--gpu-mem 0.92`，否则 8B+0.6B 两个模型的 KV 放不下 —— 这次官方对照就是栽在这里）

### 2. 本仓库的量化吞吐（价值次之，但比原估的容易）

v7 的 `QuantLinear.forward_fused()` 还没接入融合 GEMM，所以给不出自家数字。

**⭐ 关键探路结论（2026-09-27 离线摸 API 得到，省下了自写 kernel 的工作量）**：

| 方案 | 是否有现成算子 | 说明 |
|---|---|---|
| INT4 weight-only | ✅ **`torch._weight_int4pack_mm`** | 配套 `torch._convert_weight_to_int4pack` 打包；torchao/vLLM 的 `int4_weight_only` 底层用的就是它 |
| INT8 weight-only | ✅ **`torch._weight_int8pack_mm`** | per-channel scale |
| INT8 W8A8 | ✅ `torch._int_mm` | int8×int8→int32，再按 scale 反量化 |
| FP8 W8A8 | ✅ `flashinfer.gemm.mm_fp8` / `gemm_fp8_nt_groupwise` | FlashInfer 0.7 有；sm_89 支持 FP8 tensor core |
| FlashInfer 的 INT4/INT8 GEMM | ❌ **没有** | 0.7.0 里只有 fp8/fp4 系列的 GEMM |

⇒ **原本以为必须自己写 Triton dequant-GEMM（约 1 天），实际用原生 op 即可。**
估算降到 **半天（4–6 小时）**：
权重加载量化 1–2h + 接线到 `forward_fused` 1–2h + 正确性对齐 1h + 测数 1–2h。

⚠️ **两个必须先在 GPU 上确认的点**（各 10 分钟）：
1. `_weight_int4pack_mm` 在 **sm_89** 上能不能跑（tinygemm 路径理论支持 sm_80+，但需实测）
2. `group_size` 的合法取值（torchao 惯例是 32/64/128，与 v7 默认的 128 是否对得上）

⚠️ **一个容易踩的坑**：`_convert_weight_to_int4pack` 的打包布局与 v7 参考实现的
`pack_int4`（低位在前）**不是同一套约定** —— 两者必须分开存放、不能混用。

> 补任何一项前，先按 `env/pip-freeze-*.txt` 重建环境，并按第九节那条
> flash-attn wheel 文件名去匹配 —— 这一步不做，会在装依赖上先烧掉一小时。

### 3. 顺手可补的小项

- DP 多测几个负载点（现在只有 `--n-reqs 32`）
- 并发扫描补其它并行配置（现在只有 MoE TP=2 cg）
- 官方 vLLM 在 0.6B 上的对照（这次 `chain_d.log` 里全 FAIL，见第八节）

> ⚠️ 补任何一项前，先按 `env/pip-freeze-*.txt` 重建环境，并按第九节那条
> flash-attn wheel 文件名去匹配 —— 这一步不做，会在装依赖上先烧掉一小时。

## 十二、引用数字时的三条硬规矩

1. **必须说明是哪个引擎**：本仓库 v4/v6/v7，还是官方 vLLM 0.30.0。
2. **必须说明模型与规模**：同一优化在 0.6B 与 8B 上的收益方向可能相反（§六）。
3. **必须说明模式**：eager / cg / compile 之间的差距远大于引擎之间的差距（§三）。

> 这三条不是形式主义：本项目实测中，**同一份代码换规模使 TP 从"不划算"变成"+63%"**，
> **同一模型换模式使 decode 差 9.1 倍**。省掉任何一个限定词，数字都会变成误导。
