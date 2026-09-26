# 当前服务器部署

SSH 别名：`rwkv-8222`；内网地址以运维配置为准。
应用目录：`/home/chase/apps/rwkv-search-reader`。
公网 API 地址以部署配置为准；[API 使用说明](API-使用说明.md)采用示例域名。当前部署使用 HTTP，未配置 TLS。

| 进程 | 地址 | 用途 |
| --- | --- | --- |
| API | 0.0.0.0:8791 | 联网判断、任务提交、查询与 SSE |
| classifier | 127.0.0.1:18237 | G1J 1.5B + state-v6r2 分类 |
| answer | 127.0.0.1:18240 | G1J 13.3B 检索规划、阅读与回答 |
| worker | 无监听端口 | 执行 SQLite 任务队列 |

## 运行配置

两个模型固定物理 GPU 3：`CUDA_VISIBLE_DEVICES=GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0`。分类 state 为 `runs/router-state-v6r2/best-state.pth`。启动脚本为 `deploy/launch_router_state.sh` 和 `deploy/launch_answer.sh`。

回答模型目录：`/home/chase/GitHub/RWKV-LH/data/models/rwkv7-g1j-13.3b-vllm-v1`。运行时参数：`VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_RWKV7_WKV_MODE=fp32io16`；批处理 token 预算 32768，上下文 16384，并发序列 4。RWKV 关闭分块预填充，批处理预算过小可能导致长请求持续排队。

`.env.server` 使用 JSON 字符串值，权限 0600。API 与 worker 共用 `data/jobs.sqlite3`。日志和 PID 位于 `data/{api,worker,classifier,answer,supervisor}.{log,pid}`。

## 启动与维护

`deploy/supervise.py` 后台运行并拉起退出的子进程；未配置开机自启。退出 SSH 或个人电脑关机不影响服务器运行，服务器重启后需手动启动。

```bash
cd /home/chase/apps/rwkv-search-reader
nohup .venv/bin/python deploy/supervise.py >>data/supervisor.log 2>&1 </dev/null &
```

文件锁会拒绝重复实例。停止 supervisor 会停止它管理的全部子进程。更新前应暂停接收新任务并确认队列没有 queued/active 任务，再重新加载受影响进程，避免打断调用者。旧代码通过 GitHub 历史管理，不额外建立代码备份。

`/healthz` 只表示 API 存活；模型分别检查 `/v1/models`，部署验收还需真实判断与 SSE 请求。SSE 反向代理需关闭缓冲、缓存和压缩。

## 公网转发

FRP 独立运行在服务器，个人电脑不作为中转。配置为 rwkv 用户的 `/home/rwkv/frp_0.67.0_linux_amd64/search_reader_api.toml`，映射公网 18080 到服务器 127.0.0.1:8791；日志与 PID 在同目录。8222 是 SSH 端口。

以 rwkv 用户手动启动：

```bash
nohup /home/rwkv/frp_0.67.0_linux_amd64/frpc -c /home/rwkv/frp_0.67.0_linux_amd64/search_reader_api.toml >>/home/rwkv/frp_0.67.0_linux_amd64/search_reader_api.log 2>&1 </dev/null &
```

## 当前验收与限制

- 联网判断覆盖题 292/300、文档对比判断 8/8；新边界题 13/20，详见 [分类评测](search-router-v6-report.md)。
- 来源归属修复已上线，原 Python 官方文档入口问题复测正确，SSE 正常；本次 96.3 秒，仍需优化重复检索和核对开销。详见 [修复记录](source-attribution-audit.md)。
- 单次案例通过不代表所有版本归属、引用和事实准确性问题已解决。
