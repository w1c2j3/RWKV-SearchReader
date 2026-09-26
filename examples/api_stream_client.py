"""Authenticated SSE client with resumable events; API keys stay in environment."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import requests


def _session():
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + os.environ['SEARCH_READER_API_KEY']
    return session


def _base():
    return os.environ.get('SEARCH_READER_BASE_URL', 'http://47.115.88.183:18080').rstrip('/')


def submit_job(messages, locale='zh-CN', idempotency_key=None):
    """Persist/reuse idempotency_key for retries across application restarts."""
    key = idempotency_key or str(uuid.uuid4())
    deadline = time.monotonic() + 60
    with _session() as session:
        while True:
            try:
                response = session.post(_base() + '/v1/jobs',
                    json={'messages': messages, 'locale': locale},
                    headers={'Idempotency-Key': key}, timeout=(5, 20))
                if response.status_code not in (429, 502, 503, 504):
                    response.raise_for_status()
                    return response.json()
            except (requests.ConnectionError, requests.Timeout):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Submission uncertain; retry with the same Idempotency-Key: ' + key)
            time.sleep(2)


def events(job_id, last_event_id='0', wait_seconds=900):
    """Yield {id, event, data}; retain id AND displayed text for later reconnection."""
    cursor = str(last_event_id)
    deadline = time.monotonic() + wait_seconds
    with _session() as session:
        while time.monotonic() < deadline:
            try:
                with session.get(_base() + '/v1/jobs/' + job_id + '/events', stream=True,
                        headers={'Last-Event-ID': cursor}, timeout=(5, 45)) as response:
                    if response.status_code in (429, 502, 503, 504):
                        time.sleep(2)
                        continue
                    response.raise_for_status()
                    fields = {}
                    for raw in response.iter_lines(chunk_size=65536):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Stopped waiting; query job_id for its final result')
                        if raw:
                            line = raw.decode('utf-8')
                            if line.startswith(':'):
                                continue
                            name, _, value = line.partition(':')
                            if name == 'data':
                                fields[name] = fields.get(name, '') + value.lstrip(' ') + '\n'
                            else:
                                fields[name] = value.lstrip(' ')
                            continue
                        if 'event' in fields and 'data' in fields:
                            ident = fields.get('id')
                            if ident is None or int(ident) > int(cursor):
                                data = json.loads(fields['data'])
                                if ident is not None:
                                    cursor = ident
                                event = fields['event']
                                yield {'id': ident, 'event': event, 'data': data}
                                if event in ('done', 'failed'):
                                    return
                        fields = {}
                # A stream resumed after its terminal ID may close without events.
                response = session.get(_base() + '/v1/jobs/' + job_id, timeout=(5, 20))
                response.raise_for_status()
                job = response.json()
                if job['status'] in ('done', 'failed'):
                    yield {'id': None, 'event': job['status'], 'data': job}
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)
    raise TimeoutError('Stopped waiting; query job_id for its final result')


if __name__ == '__main__':
    from api_client_example import needs_search
    messages = [{'role': 'user', 'content': sys.argv[1] if len(sys.argv) > 1 else '查一下 Python 最新正式版本'}]
    if not needs_search(messages):
        print('{"needs_search": false}')  # RWKV Chat answers using its own model.
    else:
        job = submit_job(messages)
        print('job_id=' + job['job_id'], file=sys.stderr)
        if job['status'] == 'done':
            print(job['result']['content'])
        elif job['status'] == 'failed':
            raise RuntimeError(job['error'])
        else:
            for item in events(job['job_id']):
                event, data = item['event'], item['data']
                if event == 'delta':
                    print(data['content'], end='', flush=True)
                elif event == 'progress':
                    print(data['message'], file=sys.stderr)
                elif event == 'failed':
                    raise RuntimeError(data['error'])
                elif event == 'done':
                    print('\n[完成]', file=sys.stderr)
