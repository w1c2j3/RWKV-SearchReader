# RWKV SearchReader

RWKV SearchReader 是一个刻意保持简单的“先取材料、后阅读”系统。普通事实问题固定经过联网检索；上传 Markdown 后则优先阅读当前 conversation 的文档，只有用户明确要求联网或询问最新信息时才同时补充网络材料。RWKV 看不到工具定义，也不负责判断是否联网。

这是第一版可运行实现，参考了本地 `RWKV-ECRA / RWKV-Scout` 的 OpenAI 兼容 RWKV 服务配置和来源保留方式，但删除了 Planner、Tool Calling、ReAct 循环、replan 与模型路由。

## 核心行为

```text
当前 Conversation 中按顺序保存的全部消息 items
  → 最近四轮 + 与当前问题相关的更早轮次/偏好
  → Tavily Advanced Search，或当前 conversation 的 Markdown 文档片段
  → 保持 results[].content 的来源边界和相关度顺序
  → 一个网页对应一个独立 RWKV 请求，最多 10 路并发摘要
  → RWKV 审核材料是否足够；不足时生成缺口检索词并补充检索、阅读、复审
  → 每 2 份网页摘要为一组，最多 10 路 RWKV 并发归纳
  → g1i-13.3B 最终阅读器只阅读短归纳材料，直接重新回答并附带 [S1] 引用
```

这里的联网是后端行为，不是模型行为：

- `你好`、`谢谢`、`再见` 等少量完整寒暄句由后端精确匹配并直接回复，不发起搜索或模型调用；寒暄后只要带有实际问题就不会命中该规则；
- “总结前面对话”“我们刚才聊了什么”等明确操作当前聊天记录的请求，由 RWKV 直接阅读 conversation messages 回答，不会把“东西、内容”等泛称送去 Tavily；该分流由后端保守规则完成，不由模型选择工具；
- 没有附加文档的其他非空问题都必须经过检索；有文档时默认只读取文档，用户明确写出“联网、搜索、最新、今天”等要求时才合并网络材料；
- 新会话的普通单问直接使用用户原句；连续会话会先用 RWKV 把“它、这个、那次”等指代补全为独立问题，同时最终阅读器也会收到此前消息以保持偏好与语境，但 RWKV 仍不决定是否联网；
- 发给 RWKV 的 HTTP 请求不包含 `tools`、`tool_choice` 或 function schema；
- 搜索失败时请求失败，不会悄悄退回 RWKV 参数记忆；
- 任一来源摘要失败时，只让该来源退化为 Tavily 原材料摘录，其他并发任务继续；
- 搜索材料始终被标记为不可信数据，网页中的提示词不得成为模型指令。

## 已实现

- Tavily、wigolo 和 SearXNG 三种搜索后端；
- Tavily 直接采用 Advanced Search 已按查询整理的 `results[].content`；普通单问调用一次，明确的多问题按子问题分别调用；
- 首批材料完成后由 RWKV 做证据充分性审核；只有存在具体缺口时才生成 1–3 个补充检索词，每轮最多新增 10 个去重来源，默认最多补检两轮，因此来源总数不再固定为 10；
- wigolo / SearXNG 仍保留正文抓取、分块和排序作为兼容路径；
- 基础 SSRF 防护、正文大小与超时限制；
- 最多 10 路 RWKV 来源阅读并发，且每次调用只负责一个网页；
- 每个来源阅读调用只接收一个网页的 Tavily 材料，网页之间在摘要阶段完全隔离；
- 摘要异步完成后按 Tavily 原始来源编号归位；
- 每条摘要在代码层固定为 `(source_id, {title, url, published_date, summary})` 二元组，所有排序只移动整个二元组；
- Tavily 的 `published_date` 有返回时贯穿整条证据链，但检索时不设置时间限制，检索后也不会根据“今天”、“本周”等词删除来源；时间戳只作为材料交给最终 RWKV 阅读；
- 单网页摘要中出现的数字必须能在该网页材料或发布时间中找到；否则拦截该摘要并回退到 Tavily 原材料；
- 网页摘要两两分组并发归纳，减少最终 7B 模型一次读入的重复长材料；分组只压缩材料，不生成候选答案；
- 最后一次 RWKV 调用阅读并发归纳后的短材料，直接重新回答用户，不生成思维链；开放式问题按 4–8 个互不重复的信息维度组织代表性内容，避免过度压缩成少数来源，也避免按来源机械堆砌；
- `/completions` 的 Assistant 前缀固定为完整的空推理预填 `<think></think>`，所有阅读、审核和最终回答调用都直接续写公开输出；
- 最终 RWKV 直接输出自然语言答案，不使用 `facts` JSON 或固定项目符号渲染；代码只为未引用的句子补充最匹配的 `[S#]`，不改写模型原文；
- 已知百科、媒体、论坛与商店域名会标记为二手来源，供本地记录与后续分析；
- 所有 RWKV 用户提示均采用“材料在前、问题和要求在后”；
- OpenAI 兼容 RWKV `/completions` 或 `/chat/completions`；
- CLI、FastAPI、带本地历史记录和可折叠过程时间线的聊天 Web UI；
- 聊天输入框可直接上传 `.md` / `.markdown`：原文保存在本机 SQLite，按 Markdown 标题路径与段落切块，问题相关片段分别交给独立 RWKV 并发阅读；默认单文档不超过 2 MB；
- 使用 SQLite 持久化一级 `conversation` 对象及其有序 `user` / `assistant` message items；侧栏直接列 conversation，点击后读取其 items，不再从单轮日志临时拼接聊天；
- 多轮上下文不再只截最后几轮：始终保留最近四轮，并从更早对话中选回与当前问题相关的轮次和明确的用户偏好，再按原始时间顺序交给上下文阅读器；
- 支持独立的最终 RWKV 配置；当前本机使用 g1i-13.3B（16K context）承担最终综合、对话总结和文档成品生成，最终回答 `max_tokens` 统一封顶为 4090；
- 浏览器保存独立的 client ID 和当前 conversation ID；公网只能列出该 client ID 所属 conversations，不能读取其他浏览器或本机历史；旧 JSONL `session_id` 会在首次打开时一次性迁移为 conversation；
- 丰富结果接口 `/v1/answer` 和兼容接口 `/v1/chat/completions`；
- 完整 JSONL 运行 trace（默认位于 `logs/`，不记录 API Key）：成功与失败请求均保存问题、回答、来源、逐来源摘要、阶段 trace、耗时和告警；每次 RWKV 调用的完整输入、原始输出、调用阶段和耗时也保存在 `model_calls` 中，便于离线复盘；聊天数据则保存在 `logs/conversations.sqlite3`；
- 前端通过 `/v1/conversations/list` 与 `/v1/conversations/get` 恢复当前浏览器所属的完整对话；`/v1/history` 仅保留为本机运行 trace 的诊断接口，并拒绝公网读取。

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
SEARCH_READER_RWKV_SUMMARY_WORKERS=10
SEARCH_READER_REVIEW_MAX_ROUNDS=2
SEARCH_READER_REVIEW_MAX_QUERIES=3
SEARCH_READER_CONVERSATION_TURNS=10
SEARCH_READER_RWKV_MAX_TOKENS=4090

SEARCH_READER_RWKV_FINAL_BASE_URL=http://127.0.0.1:29610/v1
SEARCH_READER_RWKV_FINAL_API_KEY=rwkv-skills
SEARCH_READER_RWKV_FINAL_MODEL=rwkv7-g1i-13.3b-20260805-ctx16384
SEARCH_READER_RWKV_FINAL_PROTOCOL=completions
```

`SUMMARY_WORKERS` 同时控制逐来源阅读和分组归纳的并发。默认先读取 Tavily 前 10 条结果；审核发现证据不足时最多补检两轮，每轮最多新增 10 个来源。审核通过后再归纳材料并调用最终 RWKV 输出回答。

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

返回内容包含 `answer`、实际检索词 `queries`、`sources`、逐来源 RWKV 摘要 `evidence`、`reader_mode`、降级信息和各阶段耗时。

上传 Markdown 到当前 conversation：

```bash
curl -s http://127.0.0.1:8790/v1/documents/upload \
  -H 'Content-Type: application/json' \
  -d '{"client_id":"client_example123","conversation_id":"conv_example123","filename":"notes.md","content":"# Notes\nDocument text"}'
```

网页端的 `＋` 按钮会先创建 conversation 再上传，因此通常不需要手工调用此接口。文档绑定 conversation，后续追问会继续使用；新建聊天后不会带入旧文档。

兼容 OpenAI Chat Completions 的入口：

```bash
curl -s http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rwkv-search-reader","messages":[{"role":"user","content":"今天有哪些重要的 AI 新闻？"}]}'
```

搜索元数据位于返回对象的 `search_reader` 字段。传入多轮 `user` / `assistant` messages 时，最后一条用户消息会结合此前轮次补全指代后再检索。第一版暂不支持该兼容入口的 token 流式响应。

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

- 少量完整寒暄句走确定性本地回复；其他问题（包括翻译）仍会联网。
- 首轮普通单问使用用户原句；多轮追问使用“最近轮次 + 相关旧轮次”补全上下文指代，仍不会交给 RWKV 自主调用工具。
- 审核补检默认最多两轮，避免错误审核导致无界检索；达到边界仍不足时会保留告警并用已有材料如实回答。
- Tavily 路径只使用 Search 返回的整理材料，不继续抓整页；这有意放弃长页面的额外细节。
- 其他搜索后端的 HTML 正文提取仍是轻量实现。
- Markdown 已支持；PDF、Office、登录态网页、JavaScript 渲染和 token 级流式生成尚未实现；网页端已经可以流式显示处理阶段。
- RWKV 需要由独立的 OpenAI 兼容服务提供，本仓库不包含模型权重和推理引擎。

## 项目结构

```text
src/rwkv_search_reader/
  search.py          搜索 API 适配器
  retrieval.py       Tavily 材料保序与其他后端兼容抓取
  reader.py          RWKV 逐网页并发阅读和最终汇总输出
  pipeline.py        固定执行的主流水线
  conversations.py   Conversation 与有序 Message items 的 SQLite 持久化
  documents.py       Markdown 校验、分块与相关片段排序
  api.py             FastAPI 与兼容接口
  cli.py             命令行入口
```

## License

MIT
