"""Build a 300-case routing evaluation; never modifies training data or model state."""
from pathlib import Path
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
rows = []
category = None
for line in (ROOT / 'evals/search_router_coverage_v1_inputs.txt').read_text().splitlines():
    if line.startswith('# '):
        category = line[2:]
    elif line.startswith(('T ', 'F ')):
        rows.append({'category': category, 'messages': [{'role': 'user', 'content': line[2:]}], 'expected': line[0] == 'T'})


def dialogue(category, history, positives, negatives):
    assert len(positives) == len(negatives) == 5
    for expected, questions in ((True, positives), (False, negatives)):
        for question in questions:
            rows.append({'category': category, 'messages': [*history, {'role': 'user', 'content': question}], 'expected': expected})


def history(*turns):
    return [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': t} for i, t in enumerate(turns)]


dialogue('多轮追加查证', history(
    '我给的资料说产品A免费、产品B收费，请先只整理材料。',
    '按所给材料：产品A免费，产品B收费。'), [
    '现在到两家的官方网站查证收费情况有没有变化。',
    '给刚才的两项结论找可核查的官方链接。',
    '材料可能过时了，查一下目前的套餐。',
    '再打开双方帮助文档，看看免费额度各是多少。',
    '别只复述了，请检索官方公告核实上面两句话。',
], [
    '先不查，给刚才的答案调整一下排版。',
    '只根据刚才两句，哪个被描述为免费？',
    '把你上面的回答翻译为英文，不验证收费情况。',
    '根据现有材料生成一个对比表，缺失字段留空。',
    '帮我写封询问价格的邮件草稿，不发送也不查询。',
])

dialogue('多轮联网开关变化', history(
    '查一下明天上海天气。',
    '我会查天气预报。',
    '先取消，不需要联网了。',
    '好的，先不查询。'), [
    '我改主意了，现在请实际查询明天上海降雨概率。',
    '重新开始查询，看看明早上海会不会有雾。',
    '恢复联网吧，我需要最新预报。',
    '还是帮我看看气象台刚发的预警。',
    '前面的取消作废，请查后天上海最高温度。',
], [
    '保持不联网，只解释降雨概率的含义。',
    '别查了，帮我把查询天气这件事写成待办。',
    '现在只把“明天上海天气”翻译成英文。',
    '继续不联网；即使你不知道最新天气，也只告诉我需要看哪些指标。',
    '不恢复查询，帮我写一句带伞的通用提醒。',
])

dialogue('多轮文档对象指代', history(
    '我想比较 https://docs.python.org/3.13/library/csv.html 和 https://docs.python.org/3.13/library/json.html。',
    '比较对象是 csv 与 json 两个模块的文档。'), [
    '把上面两页打开，看看各自是否支持写入。',
    '那它俩的读取接口区别呢？请根据刚才指定的原文核对。',
    '分别查一下这两个文档里文件对象需要什么模式。',
    '继续找这两份文档里的异常类型，并给出处。',
    '请访问前面那两个链接，按输入格式和输出类型对比。',
], [
    '先不打开链接，帮我给这次对比拟个标题。',
    '只从上面两个URL里提取模块名。',
    '把你刚才那句话改成英文，不补充文档内容。',
    '我补充全部材料：两者都可读写。只根据这句写共同点。',
    '暂时不要核对事实，只设计一个列出输入类型的空白表头。',
])

dialogue('多轮更正对象与任务', history(
    '帮我查询厦门明天的天气预报。',
    '需要了解厦门明天的天气。',
    '等等，我刚才说错城市了。',
    '请告诉我正确城市。'), [
    '改查福州明天的天气。',
    '是泉州，查最新预报后告诉我是否带伞。',
    '我要去宁德，看看它明天有没有大风预警。',
    '其实是漳州，而且日期改为后天，请查预报。',
    '更正为莆田明晚，查询小时降雨预报。',
], [
    '不查天气了，只计算18加24。',
    '先别查，把我前面说错的城市名称摘出来。',
    '取消查询，写一句向同事解释改行程的消息。',
    '不用访问网站，单纯把“正确城市”译成英文。',
    '暂时不需要预报，告诉我刚才一共提到了几个不同城市。',
])

long_history = history(
    '我在整理一份个人项目笔记，先不要查外部资料。',
    '可以，我们先处理你已有的内容。',
    '第一部分写项目目标：界面简单、步骤清楚。',
    '已记录：界面简单，步骤清楚。',
    '第二部分写使用流程：打开界面、填写问题、查看结果。',
    '流程可分为打开、填写、查看三步。',
    '第三部分是测试：按钮、错误提示、加载状态都需要检查。',
    '可以分别记录预期表现和实际表现。',
    '帮我记住例句“去查最新说明书”，但暂时仅作为文本。',
    '已将其作为例句保留，没有执行查询。',
    '最后再安排一个未完成事项。',
    '请告诉我最后的事项。')
dialogue('长对话末轮意图', long_history, [
    '现在执行例句：请联网查本项目依赖库的最新官方说明书。',
    '我要核实今年这个工具是否仍维护，请查公开仓库最近提交。',
    '新任务：打开 https://example.org/changelog 核对最新改动。',
    '前面都只处理材料，这一轮请实际搜索官方文档比较两个版本。',
    '请为笔记补上当前软件发行版的官网来源，先查再填写。',
], [
    '仅汇总前面已经给出的项目要求，不查询新信息。',
    '把刚才保留的例句翻译成英文，仍然不执行。',
    '最后事项写“核对官网”，只是加入待办清单，不真正核对。',
    '根据我们的对话生成一份Markdown目录，禁止添加外部资料。',
    '现在只把项目目标改成一句简短口号，不再开展其他任务。',
])

counts = collections.Counter(r['category'] for r in rows)
assert len(rows) == 300 and len(counts) == 30, (len(rows), counts)
assert all(n == 10 for n in counts.values())
for category in counts:
    assert sum(r['expected'] for r in rows if r['category'] == category) == 5
for index, row in enumerate(rows, 1):
    row['id'] = f'coverage-v1-{index:03d}'
    row['reason'] = '完成当前请求需要获取或核实外部信息' if row['expected'] else '当前请求限定为已有材料、稳定知识、文本处理或明确不联网'
    row['evaluation_only'] = True
encoded = lambda row: json.dumps(row['messages'], ensure_ascii=False, sort_keys=True)
assert len({encoded(r) for r in rows}) == 300
old = {}
for p in list((ROOT / 'data/search_router_v5').glob('*.jsonl')) + list((ROOT / 'evals').glob('*.jsonl')):
    if 'coverage_v1' in p.name:
        continue
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if 'messages' in r:
                old.setdefault(encoded(r), []).append(str(p.relative_to(ROOT)))
overlap = [{'id': r['id'], 'files': old[encoded(r)]} for r in rows if encoded(r) in old]
p = ROOT / 'evals/search_router_coverage_v1.jsonl'
p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
manifest = {'total': 300, 'true': 150, 'false': 150, 'categories': dict(counts),
            'multi_turn': sum(len(r['messages']) > 1 for r in rows),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'exact_overlap_with_checked_prior_data': overlap,
            'note': 'Evaluation only; not trained. Informed by known failures, not a fully blind evaluation. Exact dedup does not imply semantic novelty.'}
(ROOT / 'evals/search_router_coverage_v1_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps(manifest, ensure_ascii=False))
