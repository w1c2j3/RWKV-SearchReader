# DSH-Native NoRA Adapter (`dsh-searchreader-g1l-r32`)

本目录保存 `RWKV-SearchReader` 针对 **DeepSeek Harness (`dsh-rwkv-wire-think-v2.0`)** 原生续写协议微调的 **RWKV-7 G1L 7.2B NoRA 适配器**元数据与一键自动下载合并指引（外部使用者无需自行训练）。

## Hugging Face 官方托管地址（含自动镜像回退）

- **Hugging Face 模型库**：[`chase-jay/rwkv7-g1l-dsh-searchreader-nora`](https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora)
- **直链下载**：`https://huggingface.co/chase-jay/rwkv7-g1l-dsh-searchreader-nora/resolve/main/dsh-searchreader-g1l-r32-adapter.pth`
- **备用 GitHub Releases**：`https://github.com/w1c2j3/RWKV-SearchReader/releases/download/v0.2.0-dsh-nora/dsh-searchreader-g1l-r32-adapter.pth`
- **文件大小 / SHA256**：`151,126,688` 字节（`144.12 MiB`） / `212e08c4abd2ddea24f6281b2e1ad9af3e4dc04cdfd5045e285e8be943312934`
- **适用基座模型**：`rwkv7-g1l-7.2b-20261031-ctx34304.pth`（SHA256：`ce81c2d568bc9fc8c9f7b8ddb47da070d88bf179827b311f221e1e6c5148af91`）

## 外部开发者一键自动下载并合并（零训练成本）

运行仓库内置脚本 `scripts/merge_dsh_nora.py`，它会自动按 **Hugging Face (`HF_ENDPOINT`) → `hf-mirror.com` → GitHub Releases** 顺序下载训练好的适配器、校验 SHA256，并调用 `rwkv_peft_weight_merge`（或自动回退到 PyTorch）完成 192 个线性层的无损合并：

```bash
# 国内环境可直接指定镜像加速（不指定也会自动回退 hf-mirror.com）
HF_ENDPOINT=https://hf-mirror.com uv run python scripts/merge_dsh_nora.py \
  --base-model /path/to/rwkv7-g1l-7.2b-20261031-ctx34304.pth \
  --output /path/to/rwkv7-g1l-7.2b-dsh-r32-merged.pth
```

合并完成后直接启动 `rwkv_lighting_cuda`（推荐优先使用无损 `.pth` + `RWKV_DECODE_BATCH_SIZE=160` + `RWKV_CMIX_SPARSE=off` 以保证最高事实精度与高并发吞吐）：

```bash
RWKV_MODEL_PATH=/path/to/rwkv7-g1l-7.2b-dsh-r32-merged.pth \
RWKV_DECODE_BATCH_SIZE=160 \
RWKV_CMIX_SPARSE=off \
bash deploy/launch_answer.sh
```
