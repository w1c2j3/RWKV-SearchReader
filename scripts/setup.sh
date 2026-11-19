#!/usr/bin/env bash
set -euo pipefail

# One-command full setup for RWKV-SearchReader after `git clone`:
#   1. Installs/syncs Python environment (`uv sync --dev` or `.venv` pip fallback)
#   2. Optionally fetches prebuilt `rwkv_lighting_cuda` Linux x86_64 bundle if missing
#   3. Auto-downloads RWKV-7 G1L 7.2B base checkpoint (`BlinkDL/rwkv7-g1`) +
#      trained DSH-native NoRA adapter (`chase-jay/rwkv7-g1l-dsh-searchreader-nora`)
#      with automatic `hf-mirror.com` fallback & resume, then offline-merges all 192
#      NoRA matrices into `models/rwkv7-g1l-7.2b-dsh-r32-merged.pth` (zero PyTorch needed)
#   4. Generates `.env.local` (`chmod 600`) matched to local GPU count & merged weights

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SKIP_WEIGHTS=0
SKIP_ENGINE=0
CLEANUP_BASE=0
LIGHTNING_RELEASE_TAG="${RWKV_LIGHTNING_VERSION:-v1.8.13}"
LIGHTNING_CUDA_TAG="${RWKV_LIGHTNING_CUDA_TAG:-cuda12.9}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Options:
  --skip-weights    Set up Python environment and .env.local only (skip 14.4 GB weight download/merge)
  --skip-engine     Skip attempting to download prebuilt rwkv_lighting_cuda release bundle
  --cleanup-base    Delete unmerged base .pth after verifying merged .pth to save 14.4 GB disk space
  -h, --help        Show this help message

Environment overrides:
  DEEPSEEK_API_KEY       Written directly into .env.local if provided
  HF_ENDPOINT            Hugging Face mirror (e.g. https://hf-mirror.com; auto-fallback enabled by default)
  BASE_MODEL_PATH        Reuse an existing rwkv7-g1l-7.2b-20261031-ctx34304.pth on disk
  MERGED_MODEL_PATH      Custom output path (default: models/rwkv7-g1l-7.2b-dsh-r32-merged.pth)
  RWKV_NUM_GPUS          Override detected GPU count for .env.local endpoint URLs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-weights)
      SKIP_WEIGHTS=1
      shift
      ;;
    --skip-engine)
      SKIP_ENGINE=1
      shift
      ;;
    --cleanup-base)
      CLEANUP_BASE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "==> [1/4] Preparing Python environment..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 (>= 3.10) is required." >&2
  exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "  'uv' not found; attempting install to ~/.local/bin (falls back to python3 -m venv if offline)..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf --connect-timeout 10 https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv sync --dev
  PY_BIN="${REPO_ROOT}/.venv/bin/python"
else
  echo "  Using standard python3 venv fallback (.venv)..."
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    python3 -m venv "${REPO_ROOT}/.venv"
  fi
  "${REPO_ROOT}/.venv/bin/pip" install --upgrade pip
  "${REPO_ROOT}/.venv/bin/pip" install -e . pytest httpx
  PY_BIN="${REPO_ROOT}/.venv/bin/python"
fi

echo "==> [2/4] Checking RWKV Lightning CUDA inference engine..."
HAS_ENGINE=0
if command -v rwkv_lighting_cuda >/dev/null 2>&1 \
  || [[ -x "${REPO_ROOT}/lightning/bundle/rwkv_lighting_cuda" ]] \
  || [[ -n "${RWKV_LIGHTNING_DIR:-}" && -x "${RWKV_LIGHTNING_DIR}/rwkv_lighting_cuda" ]]; then
  HAS_ENGINE=1
  echo "  [ok] rwkv_lighting_cuda detected."
elif [[ "${SKIP_ENGINE}" -eq 0 && "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  ARCHIVE_NAME="rwkv-lightning-${LIGHTNING_RELEASE_TAG}-linux-x64-${LIGHTNING_CUDA_TAG}.tar.gz"
  ARCHIVE_URL="https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/${LIGHTNING_RELEASE_TAG}/${ARCHIVE_NAME}"
  mkdir -p "${REPO_ROOT}/lightning"
  echo "  Attempting to download prebuilt engine: ${ARCHIVE_URL}"
  if "${PY_BIN}" -c "
import sys, tarfile, urllib.request
from pathlib import Path
url, dest_dir = sys.argv[1], Path(sys.argv[2])
tar_path = dest_dir / 'bundle.tar.gz'
req = urllib.request.Request(url, headers={'User-Agent': 'rwkv-search-reader/0.2'})
with urllib.request.urlopen(req, timeout=20) as r, tar_path.open('wb') as out:
    while True:
        b = r.read(4 << 20)
        if not b:
            break
        out.write(b)
with tarfile.open(tar_path, 'r:gz') as tf:
    tf.extractall(dest_dir)
tar_path.unlink(missing_ok=True)
" "${ARCHIVE_URL}" "${REPO_ROOT}/lightning" 2>/dev/null; then
    chmod +x "${REPO_ROOT}/lightning/bundle/"rwkv_* 2>/dev/null || true
    HAS_ENGINE=1
    echo "  [ok] Prebuilt rwkv_lighting_cuda extracted to lightning/bundle/"
  else
    rm -f "${REPO_ROOT}/lightning/bundle.tar.gz"
    echo "  [note] Could not auto-download prebuilt rwkv_lighting_cuda from GitHub (network/firewall)."
    echo "         If running G1L locally, grab ${ARCHIVE_NAME} from:"
    echo "         https://github.com/Alic-Li/rwkv_lightning_cuda/releases"
    echo "         and extract into ${REPO_ROOT}/lightning/ (or point RWKV_LIGHTNING_DIR to your build)."
  fi
fi

echo "==> [3/4] Preparing .env.local configuration..."
GPU_COUNT="${RWKV_NUM_GPUS:-}"
if [[ -z "${GPU_COUNT}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  else
    GPU_COUNT=1
  fi
fi
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 1 ]]; then
  GPU_COUNT=1
fi

URLS=()
for ((i = 0; i < GPU_COUNT; i++)); do
  URLS+=("http://127.0.0.1:$((18430 + i))/v1")
done
JOINED_URLS="$(IFS=','; echo "${URLS[*]}")"

if [[ ! -f "${REPO_ROOT}/.env.local" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env.local"
  chmod 600 "${REPO_ROOT}/.env.local"
  "${PY_BIN}" - "${REPO_ROOT}/.env.local" "${JOINED_URLS}" "${DEEPSEEK_API_KEY:-}" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
urls = sys.argv[2]
ds_key = sys.argv[3]

updates = {
    "SEARCH_READER_EXECUTION_MODE": "dsh_agent",
    "SEARCH_READER_RWKV_BASE_URL": urls,
    "SEARCH_READER_ROUTER_BASE_URL": urls,
    "SEARCH_READER_RWKV_MODEL": "rwkv7-g1l-7.2b-dsh-r32-merged",
    "SEARCH_READER_ROUTER_MODEL": "rwkv7-g1l-7.2b-dsh-r32-merged",
    "SEARCH_READER_RWKV_API_KEY": "",
    "SEARCH_READER_ROUTER_API_KEY": "",
}
if ds_key:
    updates["DEEPSEEK_API_KEY"] = ds_key

lines = env_path.read_text(encoding="utf-8").splitlines()
out = []
seen = set()
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        k = stripped.split("=", 1)[0].strip()
        if k in updates:
            out.append(f"{k}={updates[k]}")
            seen.add(k)
            continue
    out.append(line)
for k, v in updates.items():
    if k not in seen:
        out.append(f"{k}={v}")
env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  echo "  [ok] Created .env.local (detected ${GPU_COUNT} GPU endpoint(s): ${JOINED_URLS})"
else
  chmod 600 "${REPO_ROOT}/.env.local"
  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    "${PY_BIN}" - "${REPO_ROOT}/.env.local" "${DEEPSEEK_API_KEY}" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
ds_key = sys.argv[2]
lines = env_path.read_text(encoding="utf-8").splitlines()
out = []
replaced = False
for line in lines:
    if line.strip().startswith("DEEPSEEK_API_KEY="):
        out.append(f"DEEPSEEK_API_KEY={ds_key}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append(f"DEEPSEEK_API_KEY={ds_key}")
env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
    echo "  [ok] Updated DEEPSEEK_API_KEY in existing .env.local"
  else
    echo "  [ok] Keeping existing .env.local"
  fi
fi

if [[ "${SKIP_WEIGHTS}" -eq 0 ]]; then
  echo "==> [4/4] Downloading & offline-merging RWKV-7 G1L + DSH-native NoRA weights..."
  MERGE_ARGS=()
  if [[ "${CLEANUP_BASE}" -eq 1 ]]; then
    MERGE_ARGS+=(--cleanup-base)
  fi
  "${PY_BIN}" "${REPO_ROOT}/scripts/merge_dsh_nora.py" "${MERGE_ARGS[@]}"
else
  echo "==> [4/4] Skipped weight download/merge (--skip-weights)."
fi

cat <<EOF

====================================================================
  RWKV-SearchReader is ready!
====================================================================
  1. Ensure DEEPSEEK_API_KEY is set in .env.local
  2. Start the RWKV-7 G1L inference engine (Terminal 1):
       bash deploy/launch_answer.sh
  3. Start the SearchReader Web UI & API server (Terminal 2):
       .venv/bin/rwkv-search-reader-api
     Then open: http://127.0.0.1:8790
====================================================================
EOF
