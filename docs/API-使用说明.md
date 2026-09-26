# RWKV Chat API 使用说明

下文用 `http://api.example.com:18080` 作示例地址；接入时替换为实际服务地址。

2026-09-15 已更新服务器上的联网判断 state；以下请求、响应和流式调用方式不变。

本文中，“对方”指 RWKV Chat 调用方，“我们”指 SearchReader 服务。对方向我们发送 HTTP 请求，我们返回 JSON 或 SSE 事件流。所有业务请求携带同一个 `Authorization: Bearer YOUR_API_KEY`。

## 接入流程

```text
POST /v1/search/decision
  ├─ needs_search=false → RWKV Chat 自己回答
  └─ needs_search=true  → POST /v1/jobs
                          └─ GET /v1/jobs/{job_id}/events
                               progress → sources → delta… → done / failed
```

判断接口只判断，不搜索、不生成答案、不自动创建任务。返回 true 后由 Chat 显式提交任务；“强制联网”可直接提交任务。

推荐使用 SSE 实时展示进度和正文。原有 `GET /v1/jobs/{job_id}` 保留，可用于轮询或断线后取回完整结果。

## 1. 判断是否联网

### 对方发送请求

```http
POST /v1/search/decision HTTP/1.1
Host: api.example.com:18080
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

请求体：

```json
{
  "messages": [
    {"role": "user", "content": "介绍一下Python"},
    {"role": "assistant", "content": "Python是一种编程语言。"},
    {"role": "user", "content": "查一下最新正式版本"}
  ],
  "locale": "zh-CN"
}
```

### 我们返回响应

HTTP 200，`Content-Type: application/json`。需要联网时：

```json
{"needs_search": true}
```

不需要联网时：

```json
{"needs_search": false}
```

对方收到 true 后调用下面的创建任务接口；收到 false 后由 RWKV Chat 自己回答。判断失败为 HTTP 502，繁忙为 503；**错误不能当作 false**。

两个 POST 接口使用同一请求体。单轮只传一条 user；多轮携带历史，从 user 开始、user/assistant 交替，以 user 结束。最多 201 条（最多 100 轮历史加本轮问题），每条非空且不超过 8000 字符，总计不超过 64000 字符。不接受 system/tool 消息或附件。`locale` 默认 `zh-CN`，用于回答语言要求。

## 2. 创建搜索任务

### 对方发送请求

```http
POST /v1/jobs HTTP/1.1
Host: api.example.com:18080
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
Idempotency-Key: chat-turn-001
```

请求体与本轮联网判断的请求体一致，例如：

```json
{
  "messages": [
    {"role": "user", "content": "介绍一下Python"},
    {"role": "assistant", "content": "Python是一种编程语言。"},
    {"role": "user", "content": "查一下最新正式版本"}
  ],
  "locale": "zh-CN"
}
```

### 我们返回响应

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
Location: /v1/jobs/0123456789abcdef0123456789abcdef
Retry-After: 20
Cache-Control: no-store
```

```json
{"job_id":"0123456789abcdef0123456789abcdef","status":"running"}
```

以下接口中的 JOB_ID 替换为我们实际返回的 job_id。

保存 job_id，立即连接事件流，无需先等 20 秒。

网络重试保持同一 Idempotency-Key 和请求体，新问题换新键。键最长 128 字符，作用域为当前 API Key；相同键对应不同请求体返回 409。重复提交已结束的任务，HTTP 仍为 202，但 JSON 直接返回原有 done/failed 结果，此时不必再连接事件流。

## 3. 接收实时事件（推荐）

### 对方发送请求

此请求没有请求体，携带创建任务时的同一 API Key：

```http
GET /v1/jobs/JOB_ID/events HTTP/1.1
Host: api.example.com:18080
Authorization: Bearer YOUR_API_KEY
Accept: text/event-stream
```

### 我们持续返回响应

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache, no-transform
X-Accel-Buffering: no
```

连接保持打开。下面是连续返回的事件示例，每个事件以空行结束；data 是 JSON，id 用于断线续传。示例中来源仅展示部分字段，实际来源信息见第 4 节，sources 事件不含 content_markdown。

```text
retry: 2000

id: 101
event: progress
data: {"stage":"searching","message":"正在搜索相关资料"}

id: 102
event: sources
data: {"results":[{"id":"S1","title":"页面标题","url":"https://example.com/page","snippet":"摘要"}]}

id: 103
event: delta
data: {"content":"根据官方发布页面，"}

id: 104
event: delta
data: {"content":"该版本于……发布。"}

id: 105
event: done
data: {"job_id":"0123456789abcdef0123456789abcdef","status":"done","http_status":200,"result":{"role":"assistant","content":"根据官方发布页面，该版本于……发布。","results":[{"id":"S1","rank":1,"title":"页面标题","url":"https://example.com/page","canonical_url":"https://example.com/page","snippet":"摘要","content_markdown":"页面正文","content_status":"fetched","published_at":null,"updated_at":null,"retrieved_at":"2026-09-15T08:00:00Z","language":null}]}}

```

| 事件 | 客户端处理 |
| --- | --- |
| progress | 显示 message。stage 为 queued、searching、reading、verifying、generating |
| sources | 更新来源标题、链接和摘要；省略 content_markdown，完整正文从最终结果读取 |
| delta | 将 content 原样追加到正文，不添加额外换行或空格 |
| done | 确认完成，使用 result 保存完整答案和来源，结束连接 |
| failed | 显示错误，结束连接；已收到的正文视为未完成答案 |

阅读、整理和证据核对期间发送进度；最终生成阶段发送模型真实增量，不是生成完后拆字播放。阶段可能重复或跳过，客户端不要依赖固定顺序或百分比。通常来源先于正文；某些无需最终生成的直接返回路径可以只有进度和 done，以 done.result 为准。

已发送的正文不会在后台改写。若生成中断、达到输出上限或未通过输出校验，返回 failed，而不是把不完整正文标记为完成。正常流式任务的所有 delta.content 拼接后与 done.result.content 一致。

如果执行失败，我们返回 failed 事件并关闭流，不再发送 done：

```text
id: 105
event: failed
data: {"job_id":"0123456789abcdef0123456789abcdef","status":"failed","http_status":502,"error":{"code":"finalize","message":"任务执行失败，请稍后重试"}}

```

### 对方断线后如何请求，我们如何回应

- 空闲时约每 15 秒发送 `: heartbeat` 注释，客户端忽略其内容。
- 服务端发送 `retry: 2000`，建议断线后约 2 秒重连。
- 保存最后**已处理**的事件 id，重连时携带 `Last-Event-ID: <id>`，仅补发后续事件。ID 为整数，但不保证连续。
- 同时保留已显示正文；若正文丢失，应不带 Last-Event-ID 从头重放。不要一边保留正文、一边重复追加从头重放的事件。
- 连接已完成任务也可重放历史事件。若已处理终态 ID，再连接可能直接关闭；用查询接口确认状态即可。
- 断开 SSE 不会取消任务。服务器重启导致任务失败时，可查询或重连获取 failed。

```bash
curl -N 'http://api.example.com:18080/v1/jobs/JOB_ID/events' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Last-Event-ID: 104'
```

对方重连时发送 Last-Event-ID，我们仍返回 HTTP 200 的 SSE 流，仅重放该 ID 之后的事件。浏览器原生 EventSource 不能直接设置 Authorization 请求头，应使用支持自定义请求头的流式客户端或后端代理。

## 4. 查询完整结果／轮询兼容

### 对方发送请求

此请求没有请求体：

```http
GET /v1/jobs/JOB_ID HTTP/1.1
Host: api.example.com:18080
Authorization: Bearer YOUR_API_KEY
```

### 我们返回响应

HTTP 200，`Content-Type: application/json`。

不用 SSE 时，建议每 20 秒查询一次。任务从提交起最多执行 15 分钟，包含排队时间。查询 HTTP 200 表示成功读取任务，完成与否看 status。

进行中：

```json
{"job_id":"任务ID","status":"running"}
```

完成结果与 SSE done 的 data 相同：

```json
{
  "job_id": "任务ID",
  "status": "done",
  "http_status": 200,
  "result": {
    "role": "assistant",
    "content": "回答正文，支持Markdown和[S1]来源引用。",
    "results": [
      {
        "id": "S1",
        "rank": 1,
        "title": "页面标题",
        "url": "https://example.com/page",
        "canonical_url": "https://example.com/page",
        "snippet": "检索摘要",
        "content_markdown": "提取后的页面正文",
        "content_status": "fetched",
        "published_at": null,
        "updated_at": null,
        "retrieved_at": "2026-09-15T08:00:00Z",
        "language": null
      }
    ]
  }
}
```

`results` 可以为空。正文的 `[S1]` 对应来源 id；rank 从 1 开始。`content_status` 为 fetched 或 snippet_only，只有摘要时 content_markdown 为空字符串。正文可能截断；canonical_url 是最终抓取或规范化 URL，不保证来自 HTML canonical 标签。published_at 为 Unix 毫秒，未知为 null；updated_at、language 未可靠采集时为 null。retrieved_at 为来源记录的 UTC 时间。

任务失败，SSE failed 与查询结果使用同一结构：

```json
{
  "job_id": "任务ID",
  "status": "failed",
  "http_status": 502,
  "error": {"code": "finalize", "message": "任务执行失败，请稍后重试"}
}
```

任务超时为 http_status=504、error.code=job_timeout。done/failed 均为终态；重新执行失败任务需要新 Idempotency-Key。任务结果和事件保留 24 小时。

## 5. 请求被拒绝时，我们返回什么

例如，对方缺少或传错 API Key：

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json
WWW-Authenticate: Bearer
```

```json
{"detail":"Invalid API key"}
```


| HTTP 状态 | 含义 |
| --- | --- |
| 400 | Last-Event-ID 格式错误或超过该任务最新事件 ID |
| 401 | API Key 缺失或错误 |
| 404 | 任务不存在、已过期或不属于该 API Key |
| 409 | 幂等键已对应不同请求 |
| 422 | 请求参数不合法 |
| 429 | 提交过于频繁，按 Retry-After 重试 |
| 502 | 判断模型调用失败 |
| 503 | 服务繁忙、队列满或配置不可用 |

请求错误通常是 `{"detail":"错误信息"}`，参数校验错误的 detail 为数组。SSE 一旦建立，任务失败通过 failed 事件通知，不能再依赖 HTTP 状态。

每个 API Key 每分钟最多新建 20 个任务，全局未完成任务最多 100 个。当前不支持取消和回调。`GET /healthz` 无需认证，仅代表 API 存活。

## 6. 完整流式调用示例

下面的 Python 示例直接完成“对方发送判断请求 → 创建任务 → 读取我们返回的流式响应”，无需导入项目文件。安装 requests 后即可运行，示例已填写当前内部 API Key。

```python
import json
import requests

BASE = "http://api.example.com:18080"
HEADERS = {"Authorization": "Bearer YOUR_API_KEY"}
payload = {
    "messages": [
        {"role": "user", "content": "查询Python最新正式版本"}
    ],
    "locale": "zh-CN",
}


def read_events(response):
    """按 SSE 的空行边界解析事件，不把网络数据块当作完整 JSON。"""
    fields = {}
    for raw in response.iter_lines(chunk_size=65536):
        if raw:
            line = raw.decode("utf-8")
            if line.startswith(":"):  # 心跳注释
                continue
            name, _, value = line.partition(":")
            if name == "data":
                fields[name] = fields.get(name, "") + value.lstrip(" ") + "\n"
            else:
                fields[name] = value.lstrip(" ")
            continue
        if "event" in fields and "data" in fields:
            yield fields.get("id"), fields["event"], json.loads(fields["data"])
        fields = {}


with requests.Session() as session:
    session.headers.update(HEADERS)

    # 1. 对方发送问题，我们只返回 needs_search。
    response = session.post(BASE + "/v1/search/decision", json=payload, timeout=25)
    response.raise_for_status()  # 失败不能按 false 处理
    needs_search = response.json()["needs_search"]
    if type(needs_search) is not bool:
        raise ValueError("Invalid needs_search response")

    if not needs_search:
        print("无需联网，由 RWKV Chat 使用自己的模型回答。")
    else:
        # 2. 对方创建任务，我们返回 job_id。
        # 同一次业务请求重试时复用这个 ID；新问题必须换新 ID。
        import uuid
        request_id = str(uuid.uuid4())
        response = session.post(
            BASE + "/v1/jobs",
            json=payload,
            headers={"Idempotency-Key": request_id},
            timeout=25,
        )
        response.raise_for_status()
        job = response.json()
        job_id = job["job_id"]  # 保存到本轮聊天记录中

        if job["status"] == "done":  # 幂等重试可能直接拿到终态
            print(job["result"]["content"])
        elif job["status"] == "failed":
            raise RuntimeError(job["error"])
        else:
            # 3. 对方建立 SSE 连接，我们持续返回进度、来源和正文。
            last_event_id = "0"
            content = ""
            finished = False
            with session.get(
                BASE + f"/v1/jobs/{job_id}/events",
                headers={"Accept": "text/event-stream", "Last-Event-ID": last_event_id},
                stream=True,
                timeout=(5, 45),  # 单次连接及读取超时，不是任务总时限
            ) as response:
                response.raise_for_status()
                for event_id, event, data in read_events(response):
                    if event == "progress":
                        print("\n[进度]", data["message"])
                    elif event == "sources":
                        sources = data["results"]  # 展示标题、链接和摘要
                    elif event == "delta":
                        content += data["content"]
                        print(data["content"], end="", flush=True)
                    elif event == "done":
                        result = data["result"]  # 保存完整正文和来源
                        content = result["content"]
                        finished = True
                        print("\n[完成]")
                    elif event == "failed":
                        raise RuntimeError(data["error"])  # 将已显示正文标记为未完成

                    # 在事件处理成功后，连同已显示正文一起保存。
                    if event_id is not None:
                        last_event_id = event_id
                    if finished:
                        break

            if not finished:
                raise ConnectionError(
                    f"连接已结束但未收到终态；使用 job_id={job_id} 查询结果，"
                    f"或携带 Last-Event-ID={last_event_id} 重连并保留已有正文"
                )
```

此示例展示一次完整调用；网络异常会抛出异常。接入正式聊天流程时，按第 3 节保存 job_id、已显示正文和 Last-Event-ID，断线约 2 秒后重连，不要重新创建任务。重连后若没有事件便关闭，可调用第 4 节查询终态。从提交起最多等待 15 分钟。
