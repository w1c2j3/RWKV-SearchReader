"""Detached Linux supervisor; manually started, no boot registration."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(__file__).resolve().parents[1]
data = root / "data"
data.mkdir(exist_ok=True)
lock = (data / "supervisor.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(data / "supervisor.pid").write_text(str(os.getpid()))


def _parse_env_value(raw: str) -> str:
    val = raw.strip()
    if val.startswith('"') and val.endswith('"'):
        try:
            return str(json.loads(val))
        except ValueError:
            return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    return val


env_file = root / ".env.server"
if not env_file.exists():
    env_file = root / ".env.local"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ[key.strip()] = _parse_env_value(value)

commands = {}
if os.environ.get("SEARCH_READER_SSH_TARGET", "").strip():
    commands["ssh_forward"] = [
        str(root / ".venv/bin/python"),
        str(root / "deploy/forward_ssh.py"),
    ]
commands["api"] = [
    str(root / ".venv/bin/uvicorn"),
    "rwkv_search_reader.job_api:app",
    "--host",
    "0.0.0.0",
    "--port",
    os.environ.get("SEARCH_READER_JOB_API_PORT", "8791"),
]
commands["worker"] = [str(root / ".venv/bin/rwkv-search-reader-worker")]

router_launcher = os.environ.get("SEARCH_READER_ROUTER_LAUNCH_SCRIPT")
if router_launcher:
    commands["classifier"] = ["/bin/bash", router_launcher]
answer_launcher = os.environ.get("SEARCH_READER_ANSWER_LAUNCH_SCRIPT")
if answer_launcher:
    commands["answer"] = ["/bin/bash", answer_launcher]

stopped = False
processes = {}
logs = {}


def stop(*_):
    global stopped
    stopped = True


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, stop)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
try:
    while not stopped:
        for name, command in commands.items():
            process = processes.get(name)
            if process is None or process.poll() is not None:
                if name in logs:
                    logs[name].close()
                # Bound startup/restart logs. Pipeline traces have their own retention.
                logfile = data / (name + ".log")
                if logfile.exists() and logfile.stat().st_size > 10_000_000:
                    logfile.replace(data / (name + ".previous.log"))
                logs[name] = logfile.open("ab", buffering=0)
                processes[name] = subprocess.Popen(
                    command,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=logs[name],
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                (data / (name + ".pid")).write_text(str(processes[name].pid))
        time.sleep(3)
finally:
    for process in processes.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes.values():
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    (data / "supervisor.pid").unlink(missing_ok=True)
