"""Keep 300 examples; replace 24 training rows with missing intent boundaries."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Only training rows change. Validation/test remain fixed regression after v3.
REPLACEMENTS = {
    'research_and_learning-true-1': '检索体育锻炼对睡眠质量影响的学术研究，参考文献必须真实，不要捏造。',
    'research_and_learning-true-2': '查找关于开源软件生产效率的论文，不要凭空编作者和出处。',
    'research_and_learning-true-3': '帮我检索短视频对注意力影响的研究，没有查到的文献不要写。',
    'research_and_learning-true-4': '寻找在线教学效果的公开实验报告，请实际查找而不是编造引用。',
    'evidence_requests-true-4': '找权威资料验证鲸的分类，别只凭记忆，给实际可访问的出处。',
    'papers_and_docs-true-1': '请翻译 https://www.w3.org/standards/ 的介绍，我这里只提供了地址，没有页面内容。',
    'papers_and_docs-true-2': '把 https://www.mozilla.org/en-US/about/ 的英文介绍译成中文，正文需要你去取。',
    'linked_material-true-4': 'https://www.postgresql.org/about/ 里面写了什么？抓取英文原文后帮我翻译。',
    'colloquial_external-true-2': '我没有粘贴文章。帮我读 https://www.blender.org/about/ ，用中文概括。',
    'navigation_and_status-true-4': '现在访问 https://httpbin.org/headers ，检查实际返回结果。',
    'quoted_task_boundary-true-4': '这是我要你实际执行的任务：Search for studies about electric vehicles. 不要编造研究。',
    'daily_mixed-true-5': '我想查当前Spotify订阅费用，请看官方网站，不要给虚构的价格。',
    'papers_and_docs-false-2': '查看这段报错为什么发生：ConnectionError: connection refused at localhost:9001。',
    'papers_and_docs-false-4': '帮我分析日志：GET /api 返回404，可能有哪些常见原因？',
    'quoted_task_boundary-false-2': '翻译成中文：Look up the latest price of Apple shares.',
    'quoted_task_boundary-false-4': 'Translate into Chinese: Who is the current CEO of Microsoft?',
    'quoted_task_boundary-false-5': '这句话怎么翻译成中文：Check the latest version on the official website.',
    'colloquial_external-false-2': '帮我瞅瞅这段输出什么意思：curl: (7) Failed to connect to 192.168.1.10 port 8080。',
    'colloquial_external-false-4': '中文怎么说这句英文：Find academic papers and provide real references.',
    'navigation_and_status-false-5': '日志里写着HTTP 503 Service Unavailable，解释可能的原因。',
    'daily_mixed-false-3': '润色下面这句英语：Please search the newest phone price for me.',
    'linked_material-false-3': 'A页面内容：免费个人使用。B页面内容：收费商用。两份内容的主要区别是什么？',
    'evidence_requests-false-2': '给“查一下英伟达股票的最新价格”标注句子成分。',
    'research_and_learning-false-4': '英文句子“Search for reliable studies, do not invent citations.”的意思是什么？',
}


def main():
    source = ROOT / 'data/search_router_v3'
    output = ROOT / 'data/search_router_v4'
    output.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (source / 'train.jsonl').read_text().splitlines()]
    assert set(REPLACEMENTS) <= {row['id'] for row in rows}
    for row in rows:
        if row['id'] in REPLACEMENTS:
            row['messages'] = [{'role': 'user', 'content': REPLACEMENTS[row['id']]}]
            row['source'] = 'curated_synthetic_v4'
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    for split in ['validation', 'test']:
        (output / f'{split}.jsonl').write_bytes((source / f'{split}.jsonl').read_bytes())
    all_rows = []
    metadata = {}
    for split in ['train', 'validation', 'test']:
        path = output / f'{split}.jsonl'
        group = [json.loads(line) for line in path.read_text().splitlines()]
        all_rows.extend(group)
        metadata[split] = {'count': len(group), 'positive': sum(r['needs_search'] for r in group),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    signatures = {json.dumps(r['messages'], ensure_ascii=False, sort_keys=True) for r in all_rows}
    assert len(all_rows) == len(signatures) == 300
    assert sum(r['needs_search'] for r in all_rows) == 150
    common = [json.loads(line) for line in (ROOT / 'evals/search_router_common_v1.jsonl').read_text().splitlines()]
    assert not signatures & {json.dumps(r['messages'], ensure_ascii=False, sort_keys=True) for r in common}
    manifest = json.loads((source / 'manifest.json').read_text())
    manifest.update(splits=metadata, replaced_training_rows=list(REPLACEMENTS),
        data_source='v3 with 24 training rows replaced; total unchanged at 300',
        test_note='The fixed test was inspected after v3; v4 scores are regression, not fresh holdout.',
        selection_policy='Training screens with validation accuracy/NLL; release also compares known everyday/basic regression and selects refinement epoch 1.',
        initialization_policy='Base G1J weights frozen; refine selected v3 state after zero-state attempt underperformed.',
        training_plan={'epochs': 2, 'lr': 0.0001, 'init_state': 'runs/router-state-v3/best-state.pth', 'selection': 'validation screening, then known everyday/basic regression for release', 'selected_epoch': 1},
        stopped_attempt={'epochs_planned': 4, 'lr': 0.0007, 'init_state': None, 'reason': 'first two validation rounds 15/30 and 17/30'},
        cumulative_unique_training_examples=264)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('''# 联网判断日常数据集 v4

最终总量仍为300条，联网/不联网各150条：240训练、30验证、30固定测试。

在v3覆盖的日常场景基础上替换24条训练样本，补充：检索要求中“不要编文献”不等于禁止联网；只给URL时翻译网页仍需读取正文；对用户贴出的日志进行分析不需要联网；英文的价格/现任职位/搜索句子可以只是待翻译文本。

保留直链、网页/论文、来源核实、购买建议、多轮、天气、出行、新闻、基础知识、编程、聊天、创作等覆盖。对话材料在前，判断要求在后。不使用字符匹配。

v4零state重训尝试前两轮验证仅15/30、17/30，已停止。最终从v3最佳state继续训练2轮，学习率0.0001，仅更新time_state。训练脚本先按验证准确率和NLL保存候选，发布时再比较已知日常与基础回归题，最终选第一轮state（第二轮误联网更多）。这些回归题参与了发布选择，不是独立测试。当前训练文件240条；由于替换了24条并继承v3，累计见过264条不同训练样本，不将其宣称为从零只用当前240条。

验证和测试与v3逐字节相同；测试已在v3看过，因此v4对该集的结果只能称固定回归。100条日常题也属于已知回归集，原句未进入训练；增加单独的新场景检查，不以小样本成绩推断线上准确率。

构建脚本：scripts/build_search_router_v4_data.py。manifest.json记录替换ID、数量、校验值和来源。历史版本留存供复现，实际训练只读取本目录240条训练样本。
''')
    print(json.dumps({'total': len(all_rows), 'splits': metadata, 'replaced': len(REPLACEMENTS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
