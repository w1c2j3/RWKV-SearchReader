import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rwkv_search_reader import jobs
from rwkv_search_reader.job_api import app
from rwkv_search_reader.model_client import OpenAICompatibleClient, _ReasoningFilter
from rwkv_search_reader.models import SearchReaderError


def setup_job(tmp_path, monkeypatch):
    monkeypatch.setenv('SEARCH_READER_JOB_DB', str(tmp_path / 'jobs.db'))
    monkeypatch.setenv('SEARCH_READER_API_KEYS', 'alpha,beta')
    queue = jobs.store()
    job = queue.submit(jobs.owner_for('Bearer alpha'), {'messages': [{'role': 'user', 'content': '查询资料'}]})
    queue.claim()
    return queue, job['job_id']


def parse_events(text):
    events = []
    for block in text.split('\n\n'):
        fields = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
        if 'event' in fields:
            events.append((int(fields.get('id', '0')), fields['event'].strip(), json.loads(fields['data'])))
    return events


def test_stream_replay_auth_and_final_consistency(tmp_path, monkeypatch):
    queue, ident = setup_job(tmp_path, monkeypatch)
    queue.append_event(ident, 'delta', {'content': '中'})
    queue.append_event(ident, 'delta', {'content': '文\n回答'})
    result = {'http_status': 200, 'result': {'role': 'assistant', 'content': '中文\n回答', 'results': []}}
    queue.finish(ident, result)
    queue.append_event(ident, 'delta', {'content': 'late'})
    with TestClient(app) as client:
        url = '/v1/jobs/' + ident + '/events'
        assert client.get(url).status_code == 401
        assert client.get(url, headers={'Authorization': 'Bearer beta'}).status_code == 404
        response = client.get(url, headers={'Authorization': 'Bearer alpha'})
        assert response.headers['content-type'].startswith('text/event-stream')
        assert response.headers['x-accel-buffering'] == 'no'
        events = parse_events(response.text)
        assert [event[1] for event in events] == ['progress', 'delta', 'delta', 'done']
        assert ''.join(data['content'] for _, event, data in events if event == 'delta') == events[-1][2]['result']['content']
        resumed = client.get(url, headers={'Authorization': 'Bearer alpha', 'Last-Event-ID': str(events[1][0])})
        assert parse_events(resumed.text) == events[2:]
        assert not parse_events(client.get(url, headers={'Authorization': 'Bearer alpha', 'Last-Event-ID': str(events[-1][0])}).text)
        assert client.get(url, headers={'Authorization': 'Bearer alpha', 'Last-Event-ID': 'oops'}).status_code == 400
        assert client.get(url, headers={'Authorization': 'Bearer alpha', 'Last-Event-ID': '999999'}).status_code == 400


def test_timeout_is_recorded_once_and_events_expire(tmp_path, monkeypatch):
    queue, ident = setup_job(tmp_path, monkeypatch)
    with queue.connection() as db:
        db.execute('UPDATE jobs SET deadline=?', (time.time() - 1,))
    owner = jobs.owner_for('Bearer alpha')
    queue.events(owner, ident)
    _, events = queue.events(owner, ident)
    assert [e['event'] for e in events].count('failed') == 1
    with queue.connection() as db:
        db.execute('UPDATE jobs SET finished=?', (time.time() - 86401,))
        queue.maintenance(db, time.time())
        assert db.execute('SELECT count(*) FROM job_events').fetchone()[0] == 0


def test_reasoning_filter_split_tags():
    f = _ReasoningFilter()
    assert ''.join(f.feed(x) for x in ['<thi', 'nk>secret', '</th', 'ink>中', '文']) + f.finish() == '中文'


class Response:
    status_code = 200
    headers = {'Content-Type': 'text/event-stream'}
    def __init__(self, rows):
        self.rows = rows
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_lines(self, **kwargs):
        yield from self.rows


def test_upstream_yields_before_completion_and_rejects_disconnect():
    response = Response([b'data: {"choices":[{"text":"first"}]}', b'data: {"choices":[{"text":"second"}]}'])
    client = OpenAICompatibleClient(base_url='http://model/v1', api_key='local', model='model', timeout_seconds=10)
    def post(*args, **kwargs):
        assert kwargs['stream'] is True and kwargs['json']['stream'] is True
        return response
    client.session = SimpleNamespace(post=post)
    stream = client.stream_answer(user='question', max_tokens=100, temperature=0, stage='finalize')
    assert next(stream) == 'first'
    assert next(stream) == 'second'
    with pytest.raises(SearchReaderError, match='disconnected'):
        next(stream)


def test_upstream_chat_and_done():
    response = Response([b'data: {"choices":[{"delta":{"reasoning_content":"private"}}]}', b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}', b'data: [DONE]'])
    client = OpenAICompatibleClient(base_url='http://model/v1', api_key='local', model='model', timeout_seconds=10)
    client.session = SimpleNamespace(post=lambda *a, **kw: response)
    assert list(client.stream_answer(user='question', max_tokens=100, temperature=0, stage='finalize', protocol='chat')) == ['answer']


def test_reader_stream_does_not_rewrite_emitted_text(monkeypatch):
    from rwkv_search_reader.config import Settings
    from rwkv_search_reader.models import Evidence
    from rwkv_search_reader.reader import RWKVReader
    reader = RWKVReader(Settings())
    evidence = [Evidence(source_id='S1', title='Official', url='https://example.com', claim='版本于2024年10月7日发布', quote='2024年10月7日')]
    monkeypatch.setattr(reader, '_reduce_evidence', lambda *a, **kw: ('[S1] 版本于2024年10月7日发布。', [], 1))
    def stream_answer(**kwargs):
        assert kwargs['user'].index('[S1]') < kwargs['user'].index('问题：')
        yield '版本于2024年10月7日'
        yield '发布 [S1]。'
    client = SimpleNamespace(stream_answer=stream_answer, session=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(reader, '_client', lambda stage: client)
    received = []
    answer, warnings = reader.finalize('何时发布？', evidence, on_delta=received.append)
    assert answer == ''.join(received) == '版本于2024年10月7日发布 [S1]。'


def test_reader_invalid_stream_fails_instead_of_replacing(monkeypatch):
    from rwkv_search_reader.config import Settings
    from rwkv_search_reader.models import Evidence
    from rwkv_search_reader.reader import RWKVReader
    reader = RWKVReader(Settings())
    evidence = [Evidence(source_id='S1', title='Official', url='https://example.com', claim='有3个项目', quote='3个')]
    monkeypatch.setattr(reader, '_reduce_evidence', lambda *a, **kw: ('[S1] 有3个项目。', [], 1))
    client = SimpleNamespace(stream_answer=lambda **kw: iter(['有999个项目 [S1]。']), session=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(reader, '_client', lambda stage: client)
    received = []
    with pytest.raises(SearchReaderError, match='unsupported'):
        reader.finalize('多少个项目？', evidence, on_delta=received.append)
    assert received == ['有999个项目 [S1]。']
