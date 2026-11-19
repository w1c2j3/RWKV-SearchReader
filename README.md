# RWKV SearchReader

基于 **RWKV-7 G1L 纯续写（Pure Continuation）** 与 **DeepSeek Search & Fetch API** 的搜索问答与本地文档阅读系统，提供网页聊天界面、CLI、同步/流式接口以及面向 RWKV Chat 的异步任务队列 API。模型推理服务独立运行，本仓库不包含模型权重。

普通事实问题先检索再阅读回答；上传 Markdown 或 PDF 后，优先阅读当前会话文档，在明确要求联网或查询最新信息时补充网络材料。简单寒暄与纯对话内处理不触发外部检索。

```text
用户问题与上下文 → DSH 原生续写代理 (dsh_agent + NoRA dsh-searchreader-g1l-r32)
                 ├─ 无需外部事实 → 零工具调用直接回答 (~5s)
                 └─ 需要外部事实 → 多查询并发 web_search / 直链 web_fetch (DeepSeek API)
                                 → 9.6K Epoch 状态账本压缩 + OpenThink 核验 → 带官方链接引用答案
（同时保留 pipeline 多阶段并发分治阅读模式、本地 Markdown/PDF 会话阅读与独立 /v1/search/decision 路由接口）
```

全流程由确定性状态机或 DSH 原生续写循环驱动，**RWKV 全程使用纯续写模式**（不走 Chat Completions 模板，不依赖模型原生 Function Calling）。检索默认采用 **DeepSeek Search & Fetch API**（同时兼容 Tavily、wigolo 与 SearXNG），回答支持 Markdown 官方来源链接与 `[S1]` 编号卡片跳转。

## 安装与运行

需要 Python 3.10+、`uv`，以及可用的 RWKV-7 G1L 纯续写服务（支持 `rwkv_lighting_cuda` 的 `POST /v1/batch/completions` 或标准 `POST /v1/completions`）。

```bash
uv sync --dev --extra pdf
cp -n .env.example .env.local
chmod 600 .env.local
```

不需要 PDF 解析时可省略 `--extra pdf`。编辑 `.env.local`，填写 DeepSeek 检索密钥与 RWKV-7 G1L 端点配置（支持单端点或逗号分隔的多 GPU 端点池，自动按最少在途请求负载均衡）：

```dotenv
SEARCH_READER_SEARCH_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-key
SEARCH_READER_EXECUTION_MODE=dsh_agent

# 可选：若 G1L 部署在远端内网服务器，通过 deploy/forward_ssh.py 建立本地端口转发
SEARCH_READER_SSH_TARGET=user@your-g1l-host
SEARCH_READER_SSH_PASSWORD=your-ssh-password
SEARCH_READER_SSH_FORWARDS=18430:172.17.0.1:18430,18431:172.17.0.1:18431,18432:172.17.0.1:18432,18433:172.17.0.1:18433

# RWKV-7 G1L 纯续写端点（推荐用 rwkv_peft_weight_merge 离线合并 NoRA 权重后直接加载，单流速度翻倍）
SEARCH_READER_RWKV_BASE_URL=http://127.0.0.1:18430/v1,http://127.0.0.1:18431/v1,http://127.0.0.1:18432/v1,http://127.0.0.1:18433/v1
SEARCH_READER_RWKV_API_KEY=your-model-key
SEARCH_READER_RWKV_MODEL=rwkv7-g1l-7.2b-dsh-r32-merged
SEARCH_READER_RWKV_PROTOCOL=completions
SEARCH_READER_API_MAX_CONCURRENT_REQUESTS=16

# 联网判断路由模型（复用 G1L 多端点池并利用 Lightning Prefix Cache 加速）
SEARCH_READER_ROUTER_BASE_URL=http://127.0.0.1:18430/v1,http://127.0.0.1:18431/v1,http://127.0.0.1:18432/v1,http://127.0.0.1:18433/v1
SEARCH_READER_ROUTER_API_KEY=your-model-key
SEARCH_READER_ROUTER_MODEL=rwkv7-g1l-7.2b-dsh-r32-merged
SEARCH_READER_ROUTER_PROTOCOL=completions
```

若使用远端 GPU 服务器，可先启动内置无依赖 SSH 转发守护脚本（自动重连并带文件锁）：

```bash
uv run python deploy/forward_ssh.py
```

启动本地网页与同步 API 服务：

```bash
uv run rwkv-search-reader-api
```

访问 [网页界面](http://127.0.0.1:8790)。默认监听 `0.0.0.0:8790`；仅本机访问时设置 `SEARCH_READER_API_HOST=127.0.0.1`。其余参数说明见 [.env.example](.env.example)。

命令行直接问答：

```bash
uv run rwkv-search-reader "Python 最新稳定版的完整版本号和发布日期是什么？" --show-evidence
```

## 核心特性

- **RWKV-7 G1L OpenThink / FakeThink 分级纯续写**：与 `rwkv-pi` 共享底层 Lightning 续写规范——在单页摘要与字段提取等高速阶段采用 FakeThink（`Assistant: <think></think>\n`），在范围消歧、冲突裁决与最终合成阶段自动切换 **OpenThink**（`Assistant: <think`，`temperature=0.96, top_p=0.76`）并实时剥离思考块，同时接入原生 `POST /v1/tokens/count` 精确对齐 `ctx34304` 预算、断流自动调用 `POST /v1/server/stop` 释放请求槽位，并支持挂载 NoRA `adapter_id` / `state_id`。
- **三执行引擎与 DSH 原生协议 + 9.6K 甜点区压缩（`dsh_agent.py` / `epoch_compress.py`）**：
  - `SEARCH_READER_EXECUTION_MODE=dsh_agent`（推荐搭配专属训练的 DSH 原生 NoRA 适配器 `dsh-searchreader-g1l-r32`）：**逐字节对齐 DeepSeek Harness（`dsh-rwkv-plugin` + `tool-web`）原生续写协议**，支持多查询并发 `web_search({"queries":[...]})`、`web_fetch({"url":"..."})`、`no_tool` 直答退出、OpenThink 断句续写修复与 9.6K Epoch 状态账本折叠，训练与在线推理零转换损耗。
  - `SEARCH_READER_EXECUTION_MODE=pipeline`：多卡高并发分治阅读与程序侧证据核验管线。
  - `SEARCH_READER_EXECUTION_MODE=pi_agent`：对齐 `rwkv-pi-xml-continuation-v3` 的 `<tool_call>` 循环。
- **DeepSeek 检索与直链抓取**：默认接入 DeepSeek `/v1/tools/search`（毫秒级并发多查询、权威度 `authority` 与重排分 `rerank_score` 加权、摘要锚定正文截取）和 `/v1/tools/fetch`（指定文档 URL 直接转 Markdown 阅读）。
- **多 GPU 低延迟并发管线**：支持跨多张 GPU 最少在途请求调度、按检索对象数自适应控制阅读页面预算、原子字段完备时零额外模型审核直出，显著缩短端到端耗时。
- **会话文档阅读**：网页输入框的 `＋` 按钮支持上传 `.md`、`.markdown` 和 `.pdf`（通过 Docling 转 Markdown 并缓存），绑定当前会话隔离阅读。

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
| `POST /v1/answer` / `/v1/answer/stream` | 本地网页与 CLI 使用的同步 / 流式问答接口 |
| `GET /healthz` / `GET /readyz` | 服务存活检查 / 配置与多模型端点 TCP 可达性检查 |

## 开发与验证

```bash
uv run pytest --ignore=tests/test_pdf_evaluation.py
```

安装 `--extra pdf` 后可直接运行全量 `uv run pytest`。单元测试不依赖真实搜索密钥或在线模型；`evals/` 包含联网问答、双文档对比、联网路由分类、Markdown RAG 和 PDF 评测集：

```bash
uv run rwkv-search-reader-eval --help
uv run rwkv-search-reader-rag-eval --help
uv run rwkv-search-reader-pdf-eval --help
```

主要模块位于 `src/rwkv_search_reader/`：`pipeline.py` 编排检索与核对流程，`reader.py` 负责多端点纯续写阅读、字段绑定与校验，`model_client.py` 封装 `/v1/completions` 与 Lightning `/v1/batch/completions` 纯续写客户端，`search.py` / `retrieval.py` 实现 DeepSeek 等检索与页面抓取，`routing.py` 实现联网路由判断，`api.py` / `jobs.py` / `worker.py` 提供 HTTP 与异步队列服务。

## 限制

不支持 Office 二进制文档、登录态私有网页或本机无头浏览器 JS 渲染。检索轮次与阅读预算受配置上限约束，程序侧的数字、日期（含中英日期格式换算）与来源身份校验能拦截常见幻觉，但不能保证所有外部事实绝对无误。

最新一轮基于 **RWKV-7 G1L + DeepSeek Search** 的真实联网评测与剩余边界问题见 [真实问答质量状态](docs/quality-status.md)。

## 文档索引

- [纯续写提示词规范](PROMPTS.md)：各阶段 G1L 纯续写边界与输入输出协议。
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
