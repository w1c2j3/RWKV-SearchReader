"""Offline regressions from the captured official-document failure."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_search_reader.config import Settings
from rwkv_search_reader.models import Evidence, ResearchRecord, FieldEvidenceSelection
from rwkv_search_reader.reader import RWKVReader, _FieldFactCandidate
from rwkv_search_reader.rag_evaluation import _grade_turn, DEFAULT_DATASET

FIXTURE = Path(__file__).resolve().parents[1] / 'evals/fixtures/source-attribution/python-doc-entry.json'


def test_actual_publisher_survives_exact_subject_filter(monkeypatch):
    payload = json.loads(FIXTURE.read_text())
    evidence = [Evidence(**row) for row in payload['evidence']]
    reader = RWKVReader(replace(Settings(), rwkv_model='offline'))
    seen = set()

    def extract(record, field, item, relevance):
        seen.add(item.url)
        return None

    monkeypatch.setattr(reader, '_extract_one_field', extract)
    reader.select_evidence(payload['question'],
                           [ResearchRecord('Python官方文档', 'Python官方文档入口', ['链接'])],
                           evidence)
    assert 'https://www.w3cschool.cn/python/dict' in seen
    assert 'https://docs.python.org' in seen
    assert 'https://www.python.org/doc' in seen


def test_model_summary_cannot_become_original_evidence(monkeypatch):
    reader = RWKVReader(replace(Settings(), rwkv_model='offline'))
    item = Evidence('S1', 'Example manual', 'https://example.com/manual',
                    'The owner is Alice.', 'The owner is Bob.')
    monkeypatch.setattr(reader, '_complete', lambda *a, **kw: 'Alice')
    assert reader._extract_one_field(ResearchRecord('Example', 'Who is the owner?', ['owner']),
                                     'owner', item, 1.0) is None


def test_explicit_abstention_does_not_promote_a_single_candidate(monkeypatch):
    reader = RWKVReader(replace(Settings(), rwkv_model='offline'))
    monkeypatch.setattr(reader, '_complete', lambda *a, **kw:
                        '{"facts":[{"field":"链接","fact":"","source_ids":[]}]}')
    candidate = _FieldFactCandidate('项目文档', '链接', 'S1', 'https://mirror.example/', 1.0,
                                    url='https://mirror.example/', quote='Official manual')
    selected = reader._resolve_record_fields_together('查官方文档',
        ResearchRecord('项目文档', '查官方文档', ['链接']), {'链接': [candidate]})
    assert selected[0].status == 'missing'


@pytest.mark.parametrize('streaming', [False, True])
def test_supported_wrong_link_cannot_erase_correct_branch(monkeypatch, streaming):
    payload = json.loads(FIXTURE.read_text())
    evidence = [Evidence(**row) for row in payload['evidence']]
    reader = RWKVReader(replace(Settings(), rwkv_model='offline'))
    captured = []
    monkeypatch.setattr(reader, '_reduce_evidence', lambda *a, **kw:
                        ('Python documentation: https://docs.python.org [S7]', [], 1))
    answer = 'Python documentation: https://docs.python.org [S7]'

    def generate(user, **kwargs):
        captured.append(user)
        return answer

    monkeypatch.setattr(reader, '_complete', generate)
    client = SimpleNamespace(stream_answer=lambda **kw: iter([generate(**kw)]),
                             session=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(reader, '_client', lambda stage: client)
    chunks = []
    actual, _ = reader.finalize(payload['question'], evidence,
        research_records=[ResearchRecord('Python官方文档', '文档入口', ['链接'])],
        evidence_selections=[FieldEvidenceSelection('Python官方文档', '链接', 'S1',
                             'supported', 'https://www.w3cschool.cn/python/python-intro.html')],
        evidence_sufficient=True, on_delta=chunks.append if streaming else None)
    prompt = captured[0]
    assert 'https://docs.python.org [S7]' in prompt
    assert 'https://www.w3cschool.cn/python/dict' in prompt
    assert '候选提取，不代表官方身份或版本归属已核实' in prompt
    assert '已经逐来源核对' not in prompt
    if streaming:
        assert ''.join(chunks) == actual


@pytest.mark.parametrize('swapped', [False, True])
def test_catalog_grading_binds_values_to_product(swapped):
    data = json.loads(DEFAULT_DATASET.read_text())
    expected = next(c for c in data['cases'] if c['case_id'] == 'rag_016_catalog')['turns'][0]
    values = ['32GB、18W、2490元、两年', '64GB、35W、4290元、三年']
    if swapped:
        values.reverse()
    answer = f'Cedar Mini：{values[0]} [S1]\nCedar Pro：{values[1]} [S1]'
    result = dict(answer=answer, retrieval_mode='document', search_provider='local-markdown',
                  sources=[{'url': 'document://catalog'}], evidence=[{'source_id': 'S1'}])
    grade = _grade_turn(expected, result)
    assert grade['pass'] is (not swapped)
    assert grade['checks']['must_include_all']  # This alone passes BOTH answers.


def test_ambiguous_combined_product_row_requires_review():
    expected = {'must_bind': [{'subject': 'Mini', 'values': ['32GB']},
                             {'subject': 'Pro', 'values': ['64GB']}]}
    grade = _grade_turn(expected, {'answer': 'Mini 和 Pro 有32GB和64GB。'})
    assert not grade['content_pass']
    assert all(row['review_required'] for row in grade['binding_checks'])


def test_version_coherence_does_not_override_explicit_abstention(monkeypatch):
    reader = RWKVReader(replace(Settings(), rwkv_model='offline'))
    monkeypatch.setattr(reader, '_complete', lambda *a, **kw:
                        '{"facts":[{"field":"LTS 主版本","fact":"","source_ids":[]}]}')
    candidate = _FieldFactCandidate('Runtime', 'LTS 主版本', 'S1', '24', 1.0,
                                    lineage=(24,))
    selected = reader._resolve_record_fields_together('当前 LTS 主版本',
        ResearchRecord('Runtime', '当前 LTS', ['LTS 主版本']), {'LTS 主版本': [candidate]})
    assert selected[0].status == 'missing'
