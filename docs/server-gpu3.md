# 服务器部署、NoRA 挂载与公网转发架构

更新时间：2026-11-19。

当前默认生产架构采用 **4× NVIDIA RTX 4090 上的 RWKV-7 G1L (`rwkv7-g1l-7.2b-20261031-ctx34304`) 纯续写服务（`rwkv_lighting_cuda`）+ DSH 原生 NoRA 适配器（`dsh-searchreader-g1l-r32`）** 配合 **DeepSeek Search & Fetch API**。既支持在推理机本机运行 SearchReader，也支持通过内置 SSH 端口转发守护进程连接远端内网推理机，并支持通过 Cloudflare Quick Tunnel 一键暴露 HTTPS 公网测试入口。

## 进程与端口拓扑

| 进程 | 监听/转发地址 | 用途 |
| --- | --- | --- |
| `forward_ssh.py`（可选） | 本机 `127.0.0.1:18430..18433` → 远端 `172.17.0.1:18430..18433` | 标准库 PTY SSH 端口转发，带文件锁与断线自动重连 |
| G1L GPU 0..3（`rwkv_lighting_cuda`） | `18430`、`18431`、`18432`、`18433` | `rwkv7-g1l-7.2b-20261031-ctx34304` + `dsh-searchreader-g1l-r32` 纯续写推理（`RWKV_ADAPTER_STAGING_MIB=512`） |
| `rwkv-search-reader-api`（`uvicorn`） | `0.0.0.0:8790`（或异步队列版 `job_api` `0.0.0.0:8791`） | 网页聊天 UI、同步/流式问答、联网路由判断与异步任务 |
| `cloudflared`（可选公网联调） | `https://*.trycloudflare.com` → `http://127.0.0.1:8790` | Cloudflare 免费免域名 HTTPS 公网隧道转发（自动支持 SSE 流式输出） |
| `worker`（异步队列可选） | 无监听端口 | 领取并执行 SQLite 任务队列（多卡推荐 `SEARCH_READER_JOB_CONCURRENCY=2`） |

## 推理服务、NoRA 热注册与训练说明

远端或本机每张 GPU 运行一个 `rwkv_lighting_cuda` 实例（见 `deploy/launch_answer.sh`）：
- 参数：`RWKV_ADAPTER_STAGING_MIB=512`（确保容纳 `rank=32` 约 145 MB 的 NoRA 适配器固定显存缓冲区）与 `--chunk-size 256 --wkv32 --chunk-load --cmix-sparse off --state-db-path .../states-1843x.db`。
- NoRA 适配器注册：在各端口调用 `POST /v1/adapters`（传入 `{"adapter_id": "dsh-searchreader-g1l-r32", "path": ".../adapter-final.pth"}`），返回的 `version` 填入 `SEARCH_READER_RWKV_ADAPTER_VERSION`。
- 原生 DSH 数据构建与训练：运行 `scripts/build_dsh_nora_dataset.py` 生成零格式转换损耗的 `rwkv-lightning-segments-eos-target-v2` 数据集，再通过 `rwkv_nora_tune_dp`（`--devices 0,1,2,3 --rank 32 --alpha 32 --targets all --gemm-mode auto`）执行多卡数据并行训练。
- 接口：纯续写路由为 `POST /v1/batch/completions`（在 `dsh_agent` 模式下停止词包含 `"</tool_call>"`、`"✿"`、`"<EOD>"`、`"\nUser:"`、`"\nSystem:"`）。
- 负载均衡：`SEARCH_READER_RWKV_BASE_URL` 与 `SEARCH_READER_ROUTER_BASE_URL` 支持逗号分隔的 4 个端点，自动轮询/最少在途调度并共享 `System:` Prefix Cache。

## 配置文件与启动维护

生产环境变量保存在 `.env.server`（本地开发保存在 `.env.local`），权限必须设为 `0600`，严禁提交至 Git。模板参见 `deploy/env.server.example` 与 `.env.example`。

`deploy/supervise.py` 负责后台守护（若配置了 `SEARCH_READER_SSH_TARGET`，会自动一并拉起并监控 `deploy/forward_ssh.py`）：

```bash
cp -n deploy/env.server.example .env.server
chmod 600 .env.server
# 编辑 .env.server 填入 SEARCH_READER_API_KEYS、DEEPSEEK_API_KEY、G1L 密钥及可选 SSH 转发配置
nohup .venv/bin/python deploy/supervise.py >>data/supervisor.log 2>&1 </dev/null &
```

仅在本机开发或评测时，可单独启动 SSH 转发守护：

```bash
nohup .venv/bin/python deploy/forward_ssh.py >>logs/ssh-forward.log 2>&1 </dev/null &
```

文件锁（`data/supervisor.lock` 与 `logs/ssh-forward.lock`）防止重复拉起。更新代码前应确认 `jobs.sqlite3` 无 `queued`/`active` 任务再重启 `supervise.py`。注意远端推理机磁盘余量有限时，所有评测产物与运行日志均保存在本地仓库 `logs/` 或 `runs/`，不要向远端推理机写入大体积产物。

## 公网转发与反向代理

- **Cloudflare 免费公网隧道（快速联调体验）**：安装 `cloudflared` 后直接转发本地 `8790` 端口即可获得免证书配置的公网 `https://*.trycloudflare.com` 地址（原生透传 SSE 事件流）：
  ```bash
  cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8790
  ```
- **FRP / Nginx 固定域名部署**：若通过 FRP 暴露异步任务接口，配置映射公网端口至本机 `127.0.0.1:8791`；Nginx 反向代理示例见 `deploy/nginx.conf`。SSE 路径 `/v1/answer/stream` 与 `/v1/jobs/{id}/events` 必须关闭代理缓冲、缓存与压缩（`proxy_buffering off; proxy_cache off; gzip off;`）。生产公网环境建议前置 HTTPS/TLS 终结，避免明文传输 Bearer API Key。

## 当前验收摘要（2026-11-19）

- **DSH 原生 + `dsh-searchreader-g1l-r32` 端到端速度与准确度**：知识/数学直答题 **5.16s** 零工具调用直出；Python 3.13.0 正式发布日期与官方链接核实 **10.99s**（较原多阶段管线 34–49s 提速 3–4 倍）；`json` vs `tomllib` 标准库读写与二进制文件模式对比 **27.37s** 零警告通过（原 60.2s）。
- **联网判断路由**（G1L 7.2B 纯续写 + Prefix Cache）：`document_comparison_v1` **8/8（100%）**、`search_router` **41/42（97.6%）**、`search_router_coverage_v1` **291/300（97.0%）**。详见 [真实问答质量状态](quality-status.md)。
