# DSH-Native NoRA Adapter (`dsh-searchreader-g1l-r32`)

本目录保存 `RWKV-SearchReader` 针对 **DeepSeek Harness (`dsh-rwkv-wire-think-v2.0`)** 原生续写协议微调的 **RWKV-7 G1L 7.2B NoRA 适配器**元数据与自动下载合并指引（外部使用者无需自行训练）。

## Hugging Face 官方托管地址（含自动镜像回退）

- **NoRA 适配器模型库**：[`chase-jay/rwkv7-g1l-dsh-searchreader-nora`](https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora)
- **直链下载**：`https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora/resolve/main/dsh-searchreader-g1l-r32-adapter.pth`
- **备用 GitHub Releases**：`https://github.com/w1c2j3/RWKV-SearchReader/releases/download/v0.2.0-dsh-nora/dsh-searchreader-g1l-r32-adapter.pth`
- **适配器大小 / SHA256**：`151,126,688` 字节（`144.12 MiB`） / `212e08c4abd2ddea24f6281b2e1ad9af3e4dc04cdfd5045e285e8be943312934`
- **官方基座模型**：[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1) 中的 `rwkv7-g1l-7.2b-20261031-ctx34304.pth`（SHA256：`ce81c2d568bc9fc8c9f7b8ddb47da070d88bf179827b311f221e1e6c5148af91`）

## 一条命令自动下载并无损合并（无需 PyTorch / 无需重新训练）

在仓库根目录直接运行：

```bash
DEEPSEEK_API_KEY=sk-your-deepseek-key bash scripts/setup.sh
```

或单独运行 `scripts/merge_dsh_nora.py`（若本地尚无 `rwkv7-g1l-7.2b-20261031-ctx34304.pth`，会自动从 `BlinkDL/rwkv7-g1` / `hf-mirror.com` 断点续传下载；合并时优先调用 `rwkv_peft_weight_merge`，未安装时自动使用内置 `numpy` 流式 ZIP64 合并器完成 192 个线性层 BF16 无损合并，内存占用 `< 400 MiB`）：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/merge_dsh_nora.py
```

合并完成后直接启动 `rwkv_lighting_cuda`（默认加载 `models/rwkv7-g1l-7.2b-dsh-r32-merged.pth`，`RWKV_DECODE_BATCH_SIZE=160`、`RWKV_CMIX_SPARSE=off`）：

```bash
bash deploy/launch_answer.sh
```
