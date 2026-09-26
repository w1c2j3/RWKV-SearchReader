# 异步问答 API 与单机部署

接口使用以 [API-使用说明.md](API-使用说明.md) 为准。当前 rwkv-8222 使用原生后台 supervisor；公网地址以运维配置为准。下面 Docker Compose 是另一种部署方式。原生部署可通过 `SEARCH_READER_ROUTER_LAUNCH_SCRIPT` 与 `SEARCH_READER_ANSWER_LAUNCH_SCRIPT` 配置分类和回答模型启动脚本。

## Docker Compose 启动

Linux 服务器需要 Docker Compose；模型推理服务独立运行。本镜像不包含 PDF/Docling，任务接口只接受文本历史。

```bash
cp deploy/env.server.example .env.server
chmod 600 .env.server
# 编辑 .env.server：填写 API Key、搜索凭证和容器可访问的模型地址。
docker compose up -d --build
docker compose logs -f worker
```

入口为 `127.0.0.1:8790`，由服务器已有 HTTPS 反向代理转发至此。未配置域名、证书时不会直接暴露公网端口。容器内 `127.0.0.1` 指向容器自身，请配置实际模型服务地址。数据保存在 Compose 的 `data` 卷，不要使用 `down -v` 删除持久数据。

API 与 worker 必须共享同一个本地 SQLite 文件；不支持跨服务器共享数据库文件。一个数据库只允许一个 supervisor，通过文件锁保证。`SEARCH_READER_JOB_CONCURRENCY` 默认 1，支持 1–4；每个任务内部仍有阅读并发，先压测再提高。当前网页和同步接口不经过任务队列，部署网关只开放 `/v1/search/decision`、`/v1/jobs`、`/v1/jobs/{id}`、`/v1/jobs/{id}/events` 和 `/healthz`。不要额外将 API 容器端口直接公开。

本地开发可在配置环境变量后分别运行：

```bash
uv sync --dev
uv run rwkv-search-reader-api
uv run rwkv-search-reader-worker
```

## 失败、超时与恢复

查询任务成功时 HTTP 为 200，即使任务失败；JSON 的 http_status 是执行结果：

```json
{"job_id":"...","status":"failed","http_status":504,"error":{"code":"job_timeout","message":"任务超过 15 分钟"}}
```

15 分钟从提交开始，包含排队。执行进程有截止时间监视器，supervisor 也会终止超时进程，终态不能被迟到结果覆盖。已经发送到上游模型服务的请求是否取消由上游决定。

API 重启不影响 worker。worker 正常关闭会终止当前任务并标记失败；异常退出后子进程检测父进程消失并退出，过期心跳在 60 秒后被收敛为 `worker_lost`。排队任务保留，重启后继续领取。执行中断的任务不自动重跑，防止重复消耗搜索和推理额度；调用方可用新 Idempotency-Key 重试。

完成和失败结果保留 24 小时，worker 或接口访问时清理，过期/未知/其他调用方任务均返回 404。未配置服务端密钥返回 503，错误密钥返回 401。`/healthz` 只表示 API 存活，不代表模型或 worker 可用；上线需使用真实问题验证完整链路。

## 来源字段语义

来源 ID 沿用 pipeline 引用 ID，rank 按最终来源顺序编号。Tavily 正文与模型截断材料分别保存，只有摘要时 content_markdown 为空且状态为 snippet_only。本地抓取仍有现有的字节/文本长度限制，HTML 为提取后的纯文本（也是合法 Markdown），不保证完整页面或保留原排版。

canonical_url 优先保存本地抓取最终跳转 URL，否则回退规范化来源 URL；目前不解析 HTML canonical 标签。published_at 可解析时转毫秒，未带时区按 UTC；未知返回 null。updated_at、language 当前没有可靠采集，返回 null。retrieved_at 为来源对象创建时的 UTC 时间。无需检索的回答可以返回空 results。

## 验证

```bash
uv run pytest
# 服务器配置完成后检查 Compose 和真实问答
docker compose config -q
curl http://127.0.0.1:8790/healthz
```
