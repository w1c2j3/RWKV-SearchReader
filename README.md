# RWKV SearchReader

RWKV SearchReader 是一个刻意保持简单的“先检索、后阅读”系统。除少量完整寒暄句外，后端收到实际问题后都会固定执行联网检索；RWKV 看不到工具定义，也不负责判断是否联网。

这是第一版可运行实现，参考了本地 `RWKV-ECRA / RWKV-Scout` 的 OpenAI 兼容 RWKV 服务配置和来源保留方式，但删除了 Planner、Tool Calling、ReAct 循环、replan 与模型路由。

## 核心行为

```text
用户问题（普通单问保持原句；多个明确问号时确定性拆分）
  → Tavily Advanced Search
  → 保持 results[].content 的来源边界和相关度顺序
  → 一个网页对应一个独立 RWKV 请求，最多 10 路并发摘要
  → 每 2 份网页摘要为一组，最多 5 路 RWKV 再次并发归纳
  → 最后一个 RWKV 只阅读短归纳材料，直接重新回答并附带 [S1] 引用
```

这里的联网是后端行为，不是模型行为：

- `你好`、`谢谢`、`再见` 等少量完整寒暄句由后端精确匹配并直接回复，不发起搜索或模型调用；寒暄后只要带有实际问题就不会命中该规则；
- 其他非空问题都必须经过检索；
- 普通单问直接使用用户原句；只有多个明确问号时才按死规则拆成最多 3 个检索词，不调用模型改写或选择查询；
- 发给 RWKV 的 HTTP 请求不包含 `tools`、`tool_choice` 或 function schema；
- 搜索失败时请求失败，不会悄悄退回 RWKV 参数记忆；
- 任一来源摘要失败时，只让该来源退化为 Tavily 原材料摘录，其他并发任务继续；
- 搜索材料始终被标记为不可信数据，网页中的提示词不得成为模型指令。

## 已实现

- Tavily、wigolo 和 SearXNG 三种搜索后端；
- Tavily 直接采用 Advanced Search 已按查询整理的 `results[].content`；普通单问调用一次，明确的多问题按子问题分别调用；
- wigolo / SearXNG 仍保留正文抓取、分块和排序作为兼容路径；
- 基础 SSRF 防护、正文大小与超时限制；
- 最多 10 路 RWKV 来源阅读并发，且每次调用只负责一个网页；
- 每个来源阅读调用只接收一个网页的 Tavily 材料，网页之间在摘要阶段完全隔离；
- 摘要异步完成后按 Tavily 原始来源编号归位；
- 每条摘要在代码层固定为 `(source_id, {title, url, published_date, summary})` 二元组，所有排序只移动整个二元组；
- Tavily 的 `published_date` 有返回时贯穿整条证据链，但检索时不设置时间限制，检索后也不会根据“今天”、“本周”等词删除来源；时间戳只作为材料交给最终 RWKV 阅读；
- 单网页摘要中出现的数字必须能在该网页材料或发布时间中找到；否则拦截该摘要并回退到 Tavily 原材料；
- 网页摘要两两分组并发归纳，减少最终 7B 模型一次读入的重复长材料；分组只压缩材料，不生成候选答案；
- 最后一次 RWKV 调用阅读并发归纳后的短材料，直接重新回答用户，不生成思维链；
- 最终 RWKV 直接输出自然语言答案，不使用 `facts` JSON 或固定项目符号渲染；代码只为未引用的句子补充最匹配的 `[S#]`，不改写模型原文；
- 已知百科、媒体、论坛与商店域名会标记为二手来源，供本地记录与后续分析；
- 所有 RWKV 用户提示均采用“材料在前、问题和要求在后”；
- OpenAI 兼容 RWKV `/completions` 或 `/chat/completions`；
- CLI、FastAPI、带本地历史记录和可折叠过程时间线的聊天 Web UI；
- 丰富结果接口 `/v1/answer` 和兼容接口 `/v1/chat/completions`；
- 完整 JSONL 运行记录（默认位于 `logs/`，不记录 API Key）：成功与失败请求均保存问题、回答、来源、逐来源摘要、阶段 trace、耗时和告警；每次 RWKV 调用的完整输入、原始输出、调用阶段和耗时也保存在 `model_calls` 中，便于离线复盘；
- 本机访问时，侧栏可从 `/v1/history` 恢复最近记录；该接口拒绝公网读取，避免临时公网转发泄露历史内容。

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
```

`SUMMARY_WORKERS` 同时控制逐来源阅读和分组归纳的并发。默认会读取 Tavily 前 10 条结果，归纳为最多 5 份短材料后，再调用一次 RWKV 输出最终回答。

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

- 少量完整寒暄句走确定性本地回复；其他问题（包括翻译）仍会联网。
- 普通单问使用用户原句；包含多个明确问号时使用确定性子句拆分，仍不会交给 RWKV 自主调用工具。
- Tavily 路径只使用 Search 返回的整理材料，不继续抓整页；这有意放弃长页面的额外细节。
- 其他搜索后端的 HTML 正文提取仍是轻量实现。
- PDF、登录态网页、JavaScript 渲染和 token 级流式生成尚未实现；网页端已经可以流式显示处理阶段。
- RWKV 需要由独立的 OpenAI 兼容服务提供，本仓库不包含模型权重和推理引擎。

## 项目结构

```text
src/rwkv_search_reader/
  search.py          搜索 API 适配器
  retrieval.py       Tavily 材料保序与其他后端兼容抓取
  reader.py          RWKV 逐网页并发阅读和最终汇总输出
  pipeline.py        固定执行的主流水线
  api.py             FastAPI 与兼容接口
  cli.py             命令行入口
```

## License

MIT
