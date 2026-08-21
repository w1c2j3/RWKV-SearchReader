# RWKV SearchReader

RWKV SearchReader 是一个刻意保持简单的“先检索、后阅读”系统。每次收到用户问题，后端都会固定执行联网检索；RWKV 看不到工具定义，也不负责判断是否联网。

这是第一版可运行实现，参考了本地 `RWKV-ECRA / RWKV-Scout` 的 OpenAI 兼容 RWKV 服务配置、证据边界和来源保留方式，但删除了 Planner、Tool Calling、ReAct 循环、replan 与模型路由。

## 核心行为

```text
用户完整问题（唯一检索词）
  → 搜索 API
  → URL 去重与正文抓取
  → 固定代码分块、相关性排序
  → 付费 API 提取结构化证据（可选）
  → RWKV 阅读证据并生成带 [S1] 引用的答案
```

这里的联网是后端行为，不是模型行为：

- 每个非空问题都必须经过检索；
- 第一版直接把用户整句话作为唯一检索词，不调用模型改写查询；
- 发给 RWKV 的 HTTP 请求不包含 `tools`、`tool_choice` 或 function schema；
- 搜索失败时请求失败，不会悄悄退回 RWKV 参数记忆；
- 压缩 API 未配置或临时失败时，只退化为确定性原文摘录，不跳过检索；
- 搜索材料始终被标记为不可信数据，网页中的提示词不得成为模型指令。

## 已实现

- Tavily、wigolo 和 SearXNG 三种搜索后端；
- Tavily Extract / wigolo Fetch 优先，受限并发 HTTP 正文抓取作为补充；
- 基础 SSRF 防护、正文大小与超时限制；
- 中文二元词和英文词的确定性相关性排序；
- 可选的 OpenAI 兼容付费模型证据压缩；
- 压缩结果的来源编号校验与伪造 quote 替换；
- OpenAI 兼容 RWKV `/completions` 或 `/chat/completions`；
- CLI、FastAPI、简单 Web UI；
- 丰富结果接口 `/v1/answer` 和兼容接口 `/v1/chat/completions`；
- JSONL 运行记录（默认位于 `logs/`，不记录 API Key）。

## 安装

需要 Python 3.10+。推荐使用 `uv`：

```bash
cd /home/chase/GitHub/RWKV-SearchReader
uv sync --dev
cp .env.example .env.local
chmod 600 .env.local
```

编辑 `.env.local`。默认使用 Tavily：

```bash
SEARCH_READER_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=replace-with-your-key
# 或：TAVILY_API_KEYS=["key-1","key-2"]

SEARCH_READER_RWKV_BASE_URL=http://127.0.0.1:29613/v1
SEARCH_READER_RWKV_API_KEY=local
SEARCH_READER_RWKV_MODEL=rwkv7-g1i-13.3b-20260805-ctx16384
SEARCH_READER_RWKV_PROTOCOL=completions
```

需要付费 API 压缩材料时再配置：

```bash
SEARCH_READER_COMPRESSOR_ENABLED=1
SEARCH_READER_COMPRESSOR_BASE_URL=https://your-provider.example/v1
SEARCH_READER_COMPRESSOR_API_KEY=replace-with-your-key
SEARCH_READER_COMPRESSOR_MODEL=your-model-id
```

不配置这三个压缩字段时，系统使用确定性摘录，方便先验证搜索和 RWKV 主链路。

在当前机器上，如果新项目自己的 `.env.local` 没有 Tavily 凭证，程序会只读取相邻 `RWKV-ECRA/.env.local` 中的 `TAVILY_API_KEY` / `TAVILY_API_KEYS`，不会读取或复制其中的其他配置和密钥。密钥池会按顺序故障转移。

### 其他搜索后端

使用本地 wigolo：

```bash
SEARCH_READER_SEARCH_PROVIDER=wigolo
SEARCH_READER_WIGOLO_BASE_URL=http://127.0.0.1:3333
```

使用 SearXNG JSON API：

```bash
SEARCH_READER_SEARCH_PROVIDER=searxng
SEARCH_READER_SEARXNG_BASE_URL=http://your-searxng-host:8080
```

## 运行

命令行：

```bash
uv run rwkv-search-reader "Python 3.13.0 的正式发布日期是什么？" --show-evidence
```

启动服务和网页界面：

```bash
uv run rwkv-search-reader-api
```

默认访问 <http://127.0.0.1:8790>。

丰富结果 API：

```bash
curl -s http://127.0.0.1:8790/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"RWKV 的最新公开版本有哪些变化？"}'
```

返回内容包含 `answer`、实际检索词 `queries`、`sources`、`evidence`、降级信息和各阶段耗时。

兼容 OpenAI Chat Completions 的入口：

```bash
curl -s http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rwkv-search-reader","messages":[{"role":"user","content":"今天有哪些重要的 AI 新闻？"}]}'
```

搜索元数据位于返回对象的 `search_reader` 字段。第一版暂不支持流式响应。

## 验证

单元测试不需要真实搜索 Key 或在线模型：

```bash
uv run pytest
```

检查服务配置：

```bash
curl -s http://127.0.0.1:8790/healthz
curl -s http://127.0.0.1:8790/readyz
```

`readyz` 只验证配置是否齐全，不会消耗搜索或模型额度。

运行可断点续跑的 natural/normal 100 题测试：

```bash
uv run rwkv-search-reader-eval \
  --dataset /home/chase/GitHub/RWKV-ECRA/data/evaluation/retrieval_production_live_100_v1_20260812.json \
  --output runs/normal100-7b \
  --workers 4
```

结果写入 `results.jsonl`，汇总写入 `summary.json` 和 `REPORT.md`。再次使用同一 `--output` 会跳过已经成功记录的题目；加 `--retry-errors` 可重试失败题。该数据集没有标准答案，因此脚本报告完成率、引用率、降级率、失败阶段和耗时，不把这些结构指标冒充事实正确率。

## 第一版边界

- 所有问题都会联网，包括闲聊和翻译。这是本项目有意选择的行为。
- 只使用用户原句检索；后续可加入确定性子句拆分，但不会交给 RWKV 自主调用工具。
- HTML 正文提取是轻量实现；复杂动态页面优先使用 Tavily Extract 或 wigolo。
- PDF、登录态网页、JavaScript 渲染和流式生成尚未实现。
- RWKV 需要由独立的 OpenAI 兼容服务提供，本仓库不包含模型权重和推理引擎。

## 项目结构

```text
src/rwkv_search_reader/
  search.py          搜索 API 适配器
  retrieval.py       抓取、清洗、分块和排序
  compression.py     付费 API / 确定性证据压缩
  reader.py          无工具的 RWKV 阅读请求
  pipeline.py        固定执行的主流水线
  api.py             FastAPI 与兼容接口
  cli.py             命令行入口
```

## License

MIT
