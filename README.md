# RWKV SearchReader

基于 **RWKV-7 G1L 纯续写（Pure Continuation）** 与 **DeepSeek Search & Fetch API** 的搜索问答与本地文档阅读系统，提供网页聊天界面、CLI、同步/流式接口以及面向 RWKV Chat 的异步任务队列 API。

普通事实问题先检索再阅读回答；上传 Markdown 或 PDF 后，优先阅读当前会话文档，在明确要求联网或查询最新信息时补充网络材料。简单寒暄与纯对话内处理不触发外部检索。

```text
用户问题与上下文 → DSH 原生续写代理 (dsh_agent + 离线无损合并 NoRA rwkv7-g1l-7.2b-dsh-r32-merged)
                 ├─ 无需外部事实 → 零工具调用直接回答
                 └─ 需要外部事实 → 多查询并发 web_search / 直链 web_fetch (DeepSeek API)
                                 → 9.6K Epoch 状态账本压缩 + OpenThink 实时流式核验 → 带 [S1] 原文跳转卡片答案
（同时保留 pipeline 多阶段并发分治阅读模式、本地 Markdown/PDF 会话阅读与独立 /v1/search/decision 路由接口）
```

全流程由确定性状态机或 DSH 原生续写循环驱动，**RWKV 全程使用纯续写模式**（不走 Chat Completions 模板，不依赖模型原生 Function Calling）。检索默认采用 **DeepSeek Search & Fetch API**（同时兼容 Tavily、wigolo 与 SearXNG），网页端支持实时流式思考过程展示与 `[S1]` / `【S1】` 点击跳转到原文摘录卡片。

## 一键准备与启动（Clone 后一条命令准备好一切）

克隆仓库后，只需执行一条命令即可自动完成 **Python 环境同步（含原生 PDF 解析与纯 CPU 流式合并器，无需安装 PyTorch）→ 预编译推理引擎检测/拉取 → 自动从 Hugging Face（含 `hf-mirror.com` 自动回退与断点续传）下载官方基座权重 [`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1) 与训练好的 DSH NoRA 适配器 [`chase-jay/rwkv7-g1l-dsh-searchreader-nora`](https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora) → 192 层无损离线合并 → 生成匹配本机 GPU 数量的 `.env.local`**：

```bash
git clone https://github.com/w1c2j3/RWKV-SearchReader.git && cd RWKV-SearchReader
DEEPSEEK_API_KEY=sk-your-deepseek-key bash scripts/setup.sh
```

常用可选环境变量与参数（按需使用）：

- **国内网络加速**：未配置代理时会自动回退 `https://hf-mirror.com`，也可显式前置 `HF_ENDPOINT=https://hf-mirror.com` 跳过探测等待。
- **复用本地已有基座模型**：若磁盘已有 `rwkv7-g1l-7.2b-20261031-ctx34304.pth`，前置 `BASE_MODEL_PATH=/path/to/rwkv7-g1l-7.2b-20261031-ctx34304.pth` 即可免下载直接合并。
- **节省磁盘空间**：追加 `--cleanup-base` 可在合并校验完成后自动删除未合并的 14.4 GB 原始基座副本。
- **仅配置代码环境（远端推理/开发测试）**：`bash scripts/setup.sh --skip-weights`。

准备完成后，打开两个终端即可启动服务：

```bash
# 终端 1：启动 RWKV-7 G1L 纯续写引擎（自动加载 models/rwkv7-g1l-7.2b-dsh-r32-merged.pth 并按本机 GPU 数并发监听 18430..）
bash deploy/launch_answer.sh

# 终端 2：启动网页与 API 服务（默认高并发 256 槽位，监听 http://0.0.0.0:8790）
.venv/bin/rwkv-search-reader-api
```

访问 [网页界面](http://127.0.0.1:8790) 即可开始联网问答或上传 PDF / Markdown 阅读。仅本机访问时可在 `.env.local` 设置 `SEARCH_READER_API_HOST=127.0.0.1`。

若 G1L 部署在远端内网服务器，可在 `.env.local` 填写 `SEARCH_READER_SSH_TARGET` 后启动内置无依赖 SSH 端口转发守护进程：

```bash
.venv/bin/python deploy/forward_ssh.py
```

命令行单次问答：

```bash
.venv/bin/rwkv-search-reader "Python 最新稳定版的完整版本号和发布日期是什么？" --show-evidence
```

### 单独下载与合并 NoRA 权重说明

训练好的 DSH 原生 NoRA 适配器已公开托管于 Hugging Face [`chase-jay/rwkv7-g1l-dsh-searchreader-nora`](https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora)（同时在本仓库 [GitHub Releases `v0.2.0-dsh-nora`](https://github.com/w1c2j3/RWKV-SearchReader/releases/tag/v0.2.0-dsh-nora) 提供备份），外部使用者**无需重新训练**。也可单独调用 [scripts/merge_dsh_nora.py](scripts/merge_dsh_nora.py)（优先使用 `rwkv_peft_weight_merge`，无 C++ 工具或 PyTorch 时自动使用内置 `numpy` 流式 ZIP64 合并器，内存占用 `< 400 MiB`）：

```bash
.venv/bin/python scripts/merge_dsh_nora.py \
  --base-model models/rwkv7-g1l-7.2b-20261031-ctx34304.pth \
  --output models/rwkv7-g1l-7.2b-dsh-r32-merged.pth
```

详细权重指纹与参数说明见 [adapters/dsh-searchreader-g1l-r32/README.md](adapters/dsh-searchreader-g1l-r32/README.md) 与 [.env.example](.env.example)。

## 核心特性

- **RWKV-7 G1L OpenThink / FakeThink 分级纯续写与离线合并加速**：与 `rwkv-pi` 共享底层 Lightning 续写规范——单页摘要与字段提取采用 FakeThink，范围消歧、冲突裁决与最终合成自动切换 **OpenThink**（`Assistant: <think`，`temperature=0.96, top_p=0.76`）并向前端实时流式推送思考过程；通过 192 层离线无损合并消除动态 LoRA 投影开销（单流 `~78 tok/s`，4× RTX 4090 连续批处理峰值 `>5,600 tok/s`），支持 `x-rwkv-session-id` 会话粘性 Prefix Cache 与断流自动 `POST /v1/server/stop` 回收槽位。
- **三执行引擎与 DSH 原生协议 + 9.6K 甜点区压缩（`dsh_agent.py` / `epoch_compress.py`）**：
  - `SEARCH_READER_EXECUTION_MODE=dsh_agent`（默认，搭配 `dsh-searchreader-g1l-r32` 合并权重）：**逐字节对齐 DeepSeek Harness（`dsh-rwkv-plugin` + `tool-web`）原生续写协议**，支持多查询并发 `web_search({"queries":[...]})`、零延迟缓存复用 `web_fetch({"url":"..."})`、`no_tool` 直答退出、时效性问题自动注入当前年份锚点与 9.6K Epoch 状态账本折叠。
  - `SEARCH_READER_EXECUTION_MODE=pipeline`：多卡高并发分治阅读与程序侧证据核验管线。
  - `SEARCH_READER_EXECUTION_MODE=pi_agent`：对齐 `rwkv-pi-xml-continuation-v3` 的 `<tool_call>` 循环。
- **DeepSeek 检索与直链抓取**：默认接入 DeepSeek `/v1/tools/search`（毫秒级并发多查询、权威度 `authority` 与重排分 `rerank_score` 加权、时效性重排与摘要锚定正文截取）和 `/v1/tools/fetch`（指定文档 URL 直接转 Markdown 阅读）。
- **开箱即用的会话文档阅读与原文定位卡片**：网页输入框 `＋` 按钮支持上传 `.md`、`.markdown` 和 `.pdf`（默认内置轻量级高速 `pypdfium2` 原生文本提取，无需额外安装数 GB 依赖；若安装可选 `--extra pdf` 亦可启用 `pymupdf4llm` / `docling`），回答中的 `[S1]`、`【S1】` 引用角标可一键展开并高亮定位对应原文片段卡片。

## API 概览

交互式本地接口见 `http://127.0.0.1:8790/docs`；面向 RWKV Chat 的异步任务服务（`rwkv_search_reader.job_api:app`）详见 [API 使用说明](docs/API-使用说明.md)。

```bash
curl http://127.0.0.1:8790/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Python 最新稳定版的完整版本号和发布日期是什么？"}'
```

| 接口 | 用途 |
| --- | --- |
| `POST /v1/search/decision` | 预判对话最后一轮是否需要联网获取外部资料（纯续写路由） |
| `POST /v1/jobs` | 创建持久化异步检索回答任务（Bearer 鉴权 + 幂等键） |
| `GET /v1/jobs/{job_id}/events` | SSE 实时推送阶段进度、来源列表与纯续写增量正文 |
| `GET /v1/jobs/{job_id}` | 轮询或断线恢复完整任务结果 |
| `POST /v1/answer` / `/v1/answer/stream` | 本地网页与 CLI 使用的同步 / 流式问答接口（NDJSON 零缓冲实时流） |
| `GET /healthz` / `GET /readyz` | 服务存活检查 / 配置与多模型端点 TCP 可达性检查 |

## 开发与验证

```bash
uv run pytest
```

单元测试不依赖真实搜索密钥、PyTorch 或在线模型；`evals/` 包含联网问答、双文档对比、联网路由分类、Markdown RAG 和 PDF 评测集：

```bash
uv run rwkv-search-reader-eval --help
uv run rwkv-search-reader-rag-eval --help
uv run rwkv-search-reader-pdf-eval --help
```

主要模块位于 `src/rwkv_search_reader/`：`dsh_agent.py` 实现 DSH 原生续写循环，`pipeline.py` 编排分治检索与核对流程，`reader.py` 负责纯续写阅读与字段校验，`model_client.py` 封装 `/v1/completions` 与 Lightning `/v1/batch/completions` 客户端，`search.py` / `retrieval.py` 实现检索与页面抓取，`documents.py` 负责 PDF / Markdown 解析分块，`routing.py` 实现联网路由判断，`api.py` / `jobs.py` / `worker.py` 提供 HTTP 与异步队列服务。

## 限制

不支持 Office 二进制文档、登录态私有网页或本机无头浏览器 JS 渲染。检索轮次与阅读预算受配置上限约束，程序侧的数字、日期（含中英日期格式换算）与来源身份校验能拦截常见幻觉，但不能保证所有外部事实绝对无误。

最新一轮基于 **RWKV-7 G1L + DeepSeek Search** 的真实联网评测与剩余边界问题见 [真实问答质量状态](docs/quality-status.md)。

## 文档索引

- [纯续写提示词规范](PROMPTS.md)：各阶段 G1L 纯续写边界与输入输出协议。
- [DSH 原生 NoRA 权重说明](adapters/dsh-searchreader-g1l-r32/README.md)：Hugging Face 托管地址、SHA256 校验与无损离线合并指南。
- [API 使用说明](docs/API-使用说明.md)：RWKV Chat 接入流程、请求响应与 SSE 流式示例。
- [服务器部署与 SSH 转发](docs/server-gpu3.md)：G1L 多卡转发、`supervise.py` 守护与运维说明。
- [Docker 与异步任务运维](docs/jobs-api.md)：容器部署、SQLite 队列、超时恢复与来源字段说明。
- [真实问答质量状态](docs/quality-status.md)：U01–U10、双文档对比与联网路由最新实测记录。
- [双文档对比专项记录](docs/document-comparison-v1-report.md)。
- [联网判断路由评测记录](docs/search-router-v6-report.md)。
- [来源归属修复与验收](docs/source-attribution-audit.md)。
- [单问题调用可视化](docs/agent-demo.md)。

## License

MIT
