# DSH-Native NoRA Adapter (`dsh-searchreader-g1l-r32`)

本目录保存 `RWKV-SearchReader` 针对 **DeepSeek Harness (`dsh-rwkv-wire-think-v2.0`)** 原生续写协议微调的 **RWKV-7 G1L 7.2B NoRA 适配器**元数据与一键合并指引。

## 为什么权重放在 GitHub Releases 而不是直接 `git add`？

- 适配器文件 `dsh-searchreader-g1l-r32-adapter.pth` 体积为 **`144.1 MiB`（`151,126,688` 字节）**，超过了 GitHub 普通 Git 仓库单文件 **`100 MiB`** 的硬限制（`GH001`），且 Git LFS 免费流量配额仅 1 GB/月。
- 因此我们将完整训练权重直接发布在本仓库的 **GitHub Releases (`v0.2.0-dsh-nora`)** 中（单文件支持 2 GiB、免费不限下载带宽、克隆代码库零体积负担）：
  - 下载地址：`https://github.com/w1c2j3/RWKV-SearchReader/releases/download/v0.2.0-dsh-nora/dsh-searchreader-g1l-r32-adapter.pth`
  - 文件 SHA256：`212e08c4abd2ddea24f6281b2e1ad9af3e4dc04cdfd5045e285e8be943312934`
  - 基座模型：`rwkv7-g1l-7.2b-20261031-ctx34304.pth`（SHA256：`ce81c2d568bc9fc8c9f7b8ddb47da070d88bf179827b311f221e1e6c5148af91`）

## 外部开发者一键准备权重（3 步复现）

### 第一步：准备原版 RWKV-7 G1L 基座模型
准备好 `rwkv7-g1l-7.2b-20261031-ctx34304.pth`（未量化原始 `.pth`）。

### 第二步：一键自动下载 NoRA 并离线合并（可选同时生成 W8A16 `.rwkvq`）
运行仓库内置脚本 `scripts/merge_dsh_nora.py`，它会自动从 GitHub Releases 下载适配器、校验 SHA256，并调用 `rwkv_peft_weight_merge`（或自动回退到 PyTorch）完成 192 个线性层的无损合并：

```bash
# 仅生成合并后的无损 .pth（约 14.4 GB）
uv run python scripts/merge_dsh_nora.py \
  --base-model /path/to/rwkv7-g1l-7.2b-20261031-ctx34304.pth \
  --output /path/to/rwkv7-g1l-7.2b-dsh-r32-merged.pth

# 推荐高并发生产部署：合并后顺带量化为 7.2 GB 的 W8A16 (.rwkvq)，单张 24GB 4090 可开 --decode-batch-size 256
uv run python scripts/merge_dsh_nora.py \
  --base-model /path/to/rwkv7-g1l-7.2b-20261031-ctx34304.pth \
  --output /path/to/rwkv7-g1l-7.2b-dsh-r32-merged.pth \
  --merge-bin /path/to/rwkv_peft_weight_merge \
  --quantize-bin /path/to/rwkv_quantize \
  --quantize-w8a16 /path/to/rwkv7-g1l-7.2b-dsh-r32-merged-w8a16.rwkvq
```

### 第三步：使用 `rwkv_lighting_cuda` 直接加载并打满并发
```bash
RWKV_MODEL_PATH=/path/to/rwkv7-g1l-7.2b-dsh-r32-merged-w8a16.rwkvq \
RWKV_DECODE_BATCH_SIZE=256 \
RWKV_CMIX_SPARSE=off \
bash deploy/launch_answer.sh
```
然后在 `.env.local` 中设置 `SEARCH_READER_EXECUTION_MODE=dsh_agent`（无需再填写 `SEARCH_READER_RWKV_ADAPTER_ID`），即可获得零旁路开销与最高并发吞吐。
