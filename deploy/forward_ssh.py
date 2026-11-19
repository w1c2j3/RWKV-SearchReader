#!/usr/bin/env python3
"""Persistent SSH local port-forwarding daemon using standard library PTY.

Reads configuration from environment or `.env.local` without requiring `sshpass`.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import socket
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_forwards(spec: str) -> list[tuple[str, int, str, int]]:
    forwards: list[tuple[str, int, str, int]] = []
    for item in spec.split(","):
        part = item.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) == 3:
            bind_host = "127.0.0.1"
            local_port, remote_host, remote_port = (
                int(pieces[0]),
                pieces[1],
                int(pieces[2]),
            )
        elif len(pieces) == 4:
            bind_host, local_port, remote_host, remote_port = (
                pieces[0],
                int(pieces[1]),
                pieces[2],
                int(pieces[3]),
            )
        else:
            raise ValueError(f"Invalid forward specification: {part!r}")
        forwards.append((bind_host, local_port, remote_host, remote_port))
    return forwards


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    _load_env_file(PROJECT_ROOT / ".env.local")
    _load_env_file(PROJECT_ROOT / ".env.server")

    target = os.environ.get("SEARCH_READER_SSH_TARGET", "").strip()
    password = os.environ.get("SEARCH_READER_SSH_PASSWORD", "")
    forwards_raw = os.environ.get(
        "SEARCH_READER_SSH_FORWARDS",
        "18430:172.17.0.1:18430,18431:172.17.0.1:18431,"
        "18432:172.17.0.1:18432,18433:172.17.0.1:18433",
    ).strip()
    if not target:
        sys.stderr.write("SEARCH_READER_SSH_TARGET is not set.\n")
        return 1

    forwards = _parse_forwards(forwards_raw)
    if not forwards:
        sys.stderr.write("No SSH port forwards configured.\n")
        return 1

    lock_dir = PROJECT_ROOT / "logs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (lock_dir / "ssh-forward.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("SSH forwarder is already running.\n")
        return 0

    pid_path = lock_dir / "ssh-forward.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    stopped = False
    child_pid: int | None = None

    def handle_stop(*_: object) -> None:
        nonlocal stopped
        stopped = True
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except OSError:
                pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_stop)

    ssh_args = [
        "ssh",
        "-N",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectTimeout=10",
    ]
    for bind_host, local_port, remote_host, remote_port in forwards:
        ssh_args.extend(["-L", f"{bind_host}:{local_port}:{remote_host}:{remote_port}"])
    ssh_args.append(target)

    try:
        while not stopped:
            pid, fd = pty.fork()
            if pid == 0:
                os.execvp("ssh", ssh_args)
            child_pid = pid
            sent_password = False
            buffer = ""
            sys.stderr.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting SSH tunnel to {target} "
                f"for ports {[p[1] for p in forwards]}\n"
            )
            sys.stderr.flush()
            while not stopped:
                rlist, _, _ = select.select([fd], [], [], 1.0)
                if rlist:
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    buffer = (buffer + text)[-1024:]
                    if not sent_password and password and "password:" in buffer.lower():
                        os.write(fd, (password + "\n").encode("utf-8"))
                        sent_password = True
                        buffer = ""
                    elif sent_password and text.strip():
                        sys.stderr.write(text)
                        sys.stderr.flush()
                wpid, _ = os.waitpid(pid, os.WNOHANG)
                if wpid == pid:
                    break
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            child_pid = None
            if not stopped:
                sys.stderr.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SSH tunnel disconnected; retrying in 2s...\n"
                )
                sys.stderr.flush()
                time.sleep(2.0)
    finally:
        pid_path.unlink(missing_ok=True)
        lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
