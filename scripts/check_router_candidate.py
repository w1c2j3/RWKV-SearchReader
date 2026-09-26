"""Temporarily serve and evaluate a state on GPU 3, without changing production."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--port', type=int, default=18238)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', action='append', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    assert (run / 'best-state.pth').is_file()
    checkpoint = json.loads((run / 'config.json').read_text())['model']
    # A pre-existing service is never stopped or reused accidentally.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src'),
        SEARCH_READER_STATE_CACHE=str(ROOT / 'runs/router-state-v1/triton'),
        SEARCH_READER_ROUTER_BASE_URL=f'http://127.0.0.1:{args.port}/v1',
        SEARCH_READER_ROUTER_MODEL=args.model, SEARCH_READER_ROUTER_API_KEY='local',
        SEARCH_READER_ROUTER_PROTOCOL='completions', SEARCH_READER_ROUTER_TIMEOUT_SECONDS='20')
    command = ['/home/chase/chase/RWKV-PEFT/.venv/bin/python', str(ROOT / 'scripts/train_search_router.py'),
        '--output', str(run), '--checkpoint', checkpoint, '--serve-state', str(run / 'best-state.pth'),
        '--port', str(args.port), '--served-model-name', args.model]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (run / 'candidate-server.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Candidate exited before becoming healthy')
                try:
                    with opener.open(f'http://127.0.0.1:{args.port}/v1/models', timeout=2) as response:
                        assert json.load(response)['data'][0]['id'] == args.model
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Candidate startup timed out')
                    time.sleep(2)
            for dataset in args.dataset:
                evaluation = run.name + '-' + Path(dataset).stem
                evaluation_env = {**env, 'ROUTER_EVAL_DATASET': dataset, 'ROUTER_EVAL_NAME': evaluation}
                with (run / (Path(dataset).stem + '-evaluation.log')).open('w') as evaluation_log:
                    subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/evaluate_search_router.py')],
                        env=evaluation_env, cwd=ROOT, stdout=evaluation_log, stderr=subprocess.STDOUT, check=True, timeout=300)
                result = json.loads((ROOT / 'runs' / (evaluation + '.json')).read_text())
                print(json.dumps({'dataset': dataset, 'summary': result['summary']}, ensure_ascii=False), flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    main()
