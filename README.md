# RWKV SearchReader

基于 RWKV 的搜索问答与本地文档阅读系统，提供网页聊天界面、CLI 和 API。模型服务需独立部署，本仓库不包含权重或推理引擎。

普通事实问题先检索再回答；上传 Markdown 或 PDF 后，优先阅读当前会话的文档，明确要求联网或查询最新信息时补充网络材料。简单寒暄和对话总结不走搜索。

```text
问题与上下文 → 规划 → 搜索 / 文档分块 → 并发阅读
答案与来源引用 ← 证据归并 ← 字段绑定与材料审核（必要时补检）
```

检索由程序控制，RWKV 不调用工具。支持 Tavily（默认）、wigolo 和 SearXNG；回答保留来源引用，材料不足时给出提示。

## 安装与运行

需要 Python 3.10+、uv，以及可用的 RWKV OpenAI 兼容服务。

```bash
uv sync --dev --extra pdf
cp -n .env.example .env.local
chmod 600 .env.local
```

不需要 PDF 时可省略 `--extra pdf`。编辑 `.env.local`，填写搜索凭证和实际模型配置：

```dotenv
SEARCH_READER_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=your-tavily-key
SEARCH_READER_RWKV_BASE_URL=http://127.0.0.1:29613/v1
SEARCH_READER_RWKV_API_KEY=your-model-key
SEARCH_READER_RWKV_MODEL=your-rwkv-model
SEARCH_READER_RWKV_PROTOCOL=completions
```

仅支持 Chat Completions 的模型服务需将协议设为 `chat`。示例配置还包含独立的 `SEARCH_READER_RWKV_FINAL_*` 配置：请改成实际的最终回答模型，或删除这四项以复用主模型。

其余配置见 [.env.example](.env.example)，包括替代搜索后端、并发数、材料预算和上传限制。

```bash
uv run rwkv-search-reader-api
```

访问 [网页界面](http://127.0.0.1:8790)。默认监听 `0.0.0.0:8790`；仅本机使用时设置 `SEARCH_READER_API_HOST=127.0.0.1`。

命令行问答：

```bash
uv run rwkv-search-reader "Python 3.13.0 的正式发布日期是什么？" --show-evidence
```

## 文档与会话

- 网页输入框的 `＋` 按钮支持 `.md`、`.markdown` 和 `.pdf`。PDF 通过 Docling 转为 Markdown，扫描文档启用 OCR，相同 PDF 可复用转换缓存。
- 文档绑定当前会话，支持事实查询、全文总结和多文档对比；新建会话后不带入旧文档。
- 默认 Markdown / 转换文本上限为 10 MB，PDF 上限为 20 MB、200 页。
- 会话与文档保存在 `logs/conversations.sqlite3`，运行记录保存为 `logs/` 下的 JSONL，默认保留 30 天、总量不超过 1 GB。
- 网页显示处理阶段进度。刷新或连接断开后，已启动的回答继续执行并保存，可重新打开会话查看。

## API

请求格式及会话、文档接口见 [交互式接口文档](http://127.0.0.1:8790/docs)。

```bash
curl http://127.0.0.1:8790/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"RWKV 的最新公开版本有哪些变化？"}'
```

| 接口 | 用途 |
| --- | --- |
| `POST /v1/answer` | 返回答案、来源、证据和耗时 |
| `POST /v1/answer/stream` | SSE 处理阶段进度与最终结果 |
| `POST /v1/chat/completions` | OpenAI 兼容问答，不支持 token 流式输出 |
| `GET /healthz` | 服务存活检查 |
| `GET /readyz` | 配置及模型端点 TCP 可达性检查，不调用模型生成 |

## 开发与验证

```bash
uv run pytest
```

单元测试不需要真实搜索凭证或在线模型。`evals/` 包含搜索、Markdown RAG 和 PDF 评测数据；在线评测需要相应服务和凭证，命令参数可通过以下入口查看：

```bash
uv run rwkv-search-reader-eval --help
uv run rwkv-search-reader-rag-eval --help
uv run rwkv-search-reader-pdf-eval --help
```

主要代码位于 `src/rwkv_search_reader/`：`pipeline.py` 编排流程，`reader.py` 负责模型阅读与证据处理，`search.py` / `retrieval.py` 负责检索，`documents.py` / `conversations.py` 处理文档与会话，`api.py` 提供 HTTP 接口。网页位于 `static/index.html`。

## 限制

不支持 Office 文件、登录态网页或本机 JavaScript 渲染。检索与审核有轮次和预算上限，来源引用与校验不能保证答案完全正确。

最近一次真实联网回归的逐题分类与尚未解决的问题见 [质量状态](docs/quality-status.md)。`runs/` 和 `evals/results/` 是本地运行产物，不随代码发布。

## License

MIT

## 服务器异步任务接口

新增 `POST /v1/jobs` 提交、`GET /v1/jobs/{job_id}` 轮询，支持 API Key 隔离、SQLite 持久化与独立 worker。协议、失败处理和 Docker Compose 部署见 [异步接口与部署说明](docs/jobs-api.md)。

接入公网 API 请参阅 [API 接入文档](docs/API-使用说明.md)，包含 cURL、Python 示例和当前部署状态。

RWKV Chat 可先调用 `POST /v1/search/decision`，根据 `needs_search` 决定由 Chat 自答，还是提交 `/v1/jobs` 搜索任务。

## 文档

- [API 使用说明](docs/API-使用说明.md)：请求、响应、认证与完整流式示例。
- [当前服务器部署](docs/server-gpu3.md)：进程、端口、启动与运维。
- [Docker 与任务运维](docs/jobs-api.md)：可选容器部署、超时与恢复。
- [联网判断训练与评测](docs/search-router-v6-report.md)。
- [真实问答质量状态](docs/quality-status.md)。
- [文档对比历史基线](docs/document-comparison-v1-report.md)。
- [来源归属修复与验收](docs/source-attribution-audit.md)。
- [多智能体演示](docs/agent-demo.md)。

原始测试题维护在 `evals/`，训练数据在 `data/`，不在文档中重复展开。
