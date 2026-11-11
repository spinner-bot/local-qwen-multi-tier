# local-qwen-multi-tier

sqs_A100 服务器上的**五档位本地 Qwen 部署**：从轻量、快速、低占用，到最强但慢、独占 4 卡。

## 分级

| 级别 | 模型 | 量化 | 权重 | 卡 | 端口 | 上下文 | 定位 |
|---|---|---|---:|---|---:|---:|---|
| L1 极速 | Qwen3.5-4B | bf16 | 9.3 GB | 0 | 8101 | 32k | 冒烟、分类、路由、短摘要 |
| L2 日常 | Qwen3.5-9B | bf16 | 19.3 GB | 1 | 8102 | 32k | 问答、翻译、抽取、改写 |
| L3 质量 | Qwen3.5-27B | GPTQ-Int4 | 30.2 GB | 2 | 8103 | 16k | 代码、复杂指令、长文档 |
| L3Q 质量+ | Qwen3.5-27B | bf16 (TP2) | 55.6 GB | 2,3 | 8103 | 32k | 质量优先，替换 L4 时使用 |
| L4 吞吐 | Qwen3.5-35B-A3B | GPTQ-Int4 | 24.4 GB | 3 | 8104 | 32k | 高并发、agent、批量标注 |
| L5 旗舰 | Qwen3.5-122B-A10B | GPTQ-Int4 (TP4) | 78.9 GB | 4-7 或 0-3 | 8110 | 64k | 最难任务，慢但最强 |

权重合计约 162 GB，统一放在 `/data/Users/zrh/models/`（不入库）。

## GPU 布局

标准模式下四级并行常驻：

```
GPU0  L1 4B        :8101
GPU1  L2 9B        :8102
GPU2  L3 27B       :8103
GPU3  L4 35B-A3B   :8104
```

L5 有两种启动形态：

| 模式 | L5 用卡 | 前提 | L1–L4 |
|---|---|---|---|
| `flagship` | 4,5,6,7 | 4–7 空闲（需与 yj 协调） | 保持常驻 |
| `flagship-local` | 0,1,2,3 | 无前提 | 全部停止让路 |

> ⚠️ 4–7 当前由组内其他成员使用，且组规为每人最多 4 卡。默认使用 `flagship-local`，除非确认 4–7 已释放。

## 运行模式

```bash
scripts/llm-mode standard        # L1-L4 常驻
scripts/llm-mode flagship-local  # 停 L1-L4，L5 占 0-3
scripts/llm-mode flagship        # L5 占 4-7，L1-L4 保持
scripts/llm-mode quality         # 停 L4，27B bf16 TP2 占 2-3
scripts/llm-mode stop            # 全部停止
scripts/llm-status               # 查看状态
```

## 决策规则

1. 权重 + 目标上下文 KV ≤ 单卡 ~36 GiB → 单卡，不拆 TP；吞吐不够加副本。
2. 装不下或需要更长上下文 → TP2（L3Q）；只有 L5 用满 4 卡。
3. 160 GiB 是**单任务可用容量**，不是多进程共享内存池。
4. L1–L4 的分工是"每卡一条独立服务通道"，L5 独占时这些通道全部让路。

## 路由

LiteLLM 网关（`config/litellm.yaml`）暴露 `l1`–`l5` 别名，默认质量降级链：

```
l5 → l4 → l3 → l2 → l1
l4 → l3 → l2 → l1
l3 → l2 → l1
```

网关未部署时，客户端可直接按端口访问各档位（OpenAI 兼容接口）。

## 目录

| 路径 | 内容 | 入库 |
|---|---|---|
| `models.tsv` | 模型注册表（层级/路径/卡/端口/上下文） | 是 |
| `config/tiers.env` | 分级参数（唯一需要改的配置） | 是 |
| `config/litellm.yaml` | 网关路由与降级 | 是 |
| `scripts/llm-up` / `llm-down` / `llm-status` / `llm-mode` | 启停与状态 | 是 |
| `/data/Users/zrh/models/` | 权重 | 否 |
| `/data/Users/zrh/.envs/envs/vllm/` | 运行环境 | 否 |
| `/data/Users/zrh/run/llm/` | PID 与日志 | 否 |

## 快速开始

```bash
# 环境（一次性）
/data/Users/zrh/.envs/bin/micromamba create -p /data/Users/zrh/.envs/envs/vllm python=3.11 -y
/data/Users/zrh/.envs/envs/vllm/bin/pip install vllm "huggingface_hub[cli]"

# 下载权重（按级别顺序）
for m in Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B-GPTQ-Int4 Qwen3.5-35B-A3B-GPTQ-Int4 Qwen3.5-122B-A10B-GPTQ-Int4; do
  /data/Users/zrh/.envs/envs/vllm/bin/hf download "Qwen/$m" --local-dir "/data/Users/zrh/models/$m"
done

# 起服务
scripts/llm-mode standard
scripts/llm-status
```

## 安全与共享

- 服务一律绑定 `127.0.0.1`，外部访问走 SSH 隧道：`ssh -L 8101:127.0.0.1:8101 sqs_A100`
- API key 放 `/data/Users/zrh/.secrets/llm_api_key`（600 权限），不入库
- 启停脚本只操作自己的 PID 文件，不按进程名 kill，不碰他人任务
- 共享 root 环境：任何文件对同机成员可读，勿在仓库中放凭据
