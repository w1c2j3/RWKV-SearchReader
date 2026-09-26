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
for line in (root / ".env.server").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        key, value = line.split("=", 1)
        os.environ[key] = json.loads(value)
commands = {
    "api": [str(root / ".venv/bin/uvicorn"), "rwkv_search_reader.job_api:app", "--host", "0.0.0.0", "--port", "8791"],
    "worker": [str(root / ".venv/bin/rwkv-search-reader-worker")],
}
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
                processes[name] = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=logs[name], stderr=subprocess.STDOUT, start_new_session=True)
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
