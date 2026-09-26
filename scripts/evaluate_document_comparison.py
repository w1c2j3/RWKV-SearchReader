"""Public comparison eval: route first, then actual SSE jobs; human quality review follows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from api_client_example import needs_search
from api_stream_client import submit_job, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='evals/document_comparison_v1.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--force-diagnostic-jobs', action='store_true', help='Explicitly test retrieval despite a false routing decision; never changes production routing')
    args = parser.parse_args()
    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
              'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cases': []}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for case in rows:
        row = {**case, 'quality_verdict': 'pending_manual_review' if case['run_job'] else 'not_applicable'}
        start = time.monotonic()
        try:
            row['decision'] = needs_search(case['messages'])
            row['decision_correct'] = row['decision'] is case['expected']
        except Exception as exc:
            row['decision_error'] = type(exc).__name__
            row['decision_correct'] = False
        row['decision_seconds'] = round(time.monotonic() - start, 3)
        report['cases'].append(row)
        save()
        print(case['id'], 'decision', row.get('decision'), 'expected', case['expected'], flush=True)
    for row in report['cases']:
        if not row['run_job']:
            continue
        row['route_bypassed_for_diagnostic'] = row.get('decision') is not True and args.force_diagnostic_jobs
        if row.get('decision') is not True and not args.force_diagnostic_jobs:
            row['job_status'] = 'not_submitted_decision_did_not_allow_search'
            save()
            continue
        start = time.monotonic()
        parts = []
        terminal = None
        row['delta_count'] = 0
        row['event_count'] = 0
        try:
            job = submit_job(row['messages'], idempotency_key=out.name + '-' + row['id'])
            row['job_id'] = job['job_id']
            print(row['id'], 'submitted', job['job_id'], flush=True)
            save()
            if job['status'] in ('done', 'failed'):
                terminal = job
            else:
                for event in events(job['job_id']):
                    row['event_count'] += 1
                    if event['event'] == 'delta':
                        row.setdefault('first_delta_seconds', round(time.monotonic() - start, 3))
                        row['delta_count'] += 1
                        parts.append(event['data']['content'])
                    elif event['event'] == 'progress':
                        print(row['id'], round(time.monotonic() - start), event['data']['stage'], flush=True)
                    elif event['event'] in ('done', 'failed'):
                        terminal = event['data']
            if terminal is None:
                raise RuntimeError('Missing terminal event')
            (out / (row['id'] + '.json')).write_text(json.dumps(terminal, ensure_ascii=False, indent=2))
            row['job_status'] = terminal['status']
            row['seconds'] = round(time.monotonic() - start, 3)
            row['partial_content'] = ''.join(parts) if terminal['status'] != 'done' else ''
            row['error'] = terminal.get('error')
            if terminal['status'] == 'done':
                result = terminal['result']
                row['answer'] = result['content']
                row['sources'] = [{k: source[k] for k in ('id', 'title', 'url', 'content_status')} for source in result['results']]
                row['stream_consistent'] = ''.join(parts) == result['content'] if parts else None
            print(row['id'], row['job_status'], row['seconds'], 'seconds', 'deltas', row['delta_count'], flush=True)
        except Exception as exc:
            row['job_status'] = 'client_error'
            row['client_error'] = type(exc).__name__
            row['partial_content'] = ''.join(parts)
            print(row['id'], 'client_error', type(exc).__name__, flush=True)
        save()
    report['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save()


if __name__ == '__main__':
    main()
