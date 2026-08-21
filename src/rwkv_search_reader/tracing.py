from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def append_trace(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"runs-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
