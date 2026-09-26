"""Build 300 curated everyday routing examples, with fixed labels before training."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dialogue(user, assistant, followup):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant},
            {"role": "user", "content": followup}]


# Each group contains five external-information requests and five offline controls.
TRAIN = {
    "linked_material": ([
        "阅读 https://github.com/psf/requests 的介绍，说明它怎么安装。",
        "看看 https://www.djangoproject.com/start/ 里面的上手流程，整理一下。",
        "https://www.sqlite.org/about.html 这篇介绍的重点是什么？",
        "把 https://www.postgresql.org/about/ 页面正文翻译成中文。",
        "打开 https://www.ruby-lang.org/en/ 和 https://www.perl.org/ ，比较两站首页介绍。",
    ], [
        "README全文在这里：运行pip install demo，然后执行demo start。概括安装流程。",
        "把这段网页正文翻译成中文：Install the package and start the server.",
        "这里是两个页面的全文，A：免费、仅限个人。B：付费、可商用。仅根据这些文字比较。",
        "这篇文章网址 https://example.com/a ，但你不用打开。帮我把网址写成Markdown链接。",
        "只根据粘贴的段落总结，不读链接：本服务周末关闭，工作日开放。来源 https://example.com。",
    ]),
    "papers_and_docs": ([
        "帮我阅读 https://arxiv.org/abs/1706.03762 ，用中文提炼方法。",
        "这个PDF讲了哪些实验：https://arxiv.org/pdf/1810.04805 ？",
        "查阅 https://docs.python.org/3/library/pathlib.html ，列出页面提到的文件操作方法。",
        "读 https://nginx.org/en/docs/ 的说明，整理文档入口。",
        "到 https://github.com/redis/redis/releases 看看新发布的说明并总结。",
    ], [
        "论文摘要已经贴全：我们比较三种排序算法，发现输入分布影响耗时。提取研究问题。",
        "如何用Python从PDF提取文字？给一般代码示例，不必查文档。",
        "我上传的文字摘录如下：路径对象支持拼接、判断文件是否存在。把这两点整理成清单。",
        "解释什么是README文件。",
        "帮我润色这句发布说明：修好了启动失败的问题。",
    ]),
    "evidence_requests": ([
        "核对一下人类第一次登月是哪一天，给我可靠出处。",
        "我要确认 Go 1.22 的发布日期，找官方公告作为依据。",
        "找权威来源验证珠穆朗玛峰现在公布的高度。",
        "有人说鲸不是鱼，帮我找科普机构的资料确认。",
        "给出中国第七次人口普查人口总数及官方原文链接。",
    ], [
        "按给定材料回答：公告写明产品于2024年4月2日发布。发布日期是哪天？",
        "什么叫一手资料和二手资料？",
        "不联网，解释为什么鲸属于哺乳动物。",
        "把“请提供官方依据”这句话改得礼貌些。",
        "仅检查这段论证的逻辑：所有猫都是动物，小白是猫，所以小白是动物。",
    ]),
    "research_and_learning": ([
        "检索睡眠时长和记忆力关系的研究，列出论文题目。",
        "给我找几篇讲LoRA的原始论文及来源。",
        "我想学习Linux，搜一些适合新手的免费课程。",
        "寻找比较Transformer与RNN的公开实验报告。",
        "查找关于海洋塑料污染的综述，附可访问的全文入口。",
    ], [
        "帮我拟定睡眠研究的检索词，不要实际检索。",
        "解释什么是综述论文。",
        "给我设计一个七天Linux基础学习计划，不需要链接或推荐课程。",
        "以下两篇摘要都已给出：A研究睡眠，B研究运动。请按主题分类。",
        "为“海洋污染”这个研究题目写一个假设，不要编文献。",
    ]),
    "buying_and_services": ([
        "预算三千元想换手机，现在有什么合适的型号？",
        "想买一台适合写代码的轻薄本，帮我比较现在能买到的几款。",
        "现在哪家云厂商的对象存储更划算，看看具体价目表。",
        "最近准备买扫地机器人，有什么值得买的？",
        "今年新出的运动耳机有哪些，帮我选一款。",
    ], [
        "选购笔记本时CPU、内存和硬盘分别影响什么？不推荐具体机型。",
        "只按我给的信息推荐：手机A续航10小时3000元，B续航8小时2500元，我更重视续航。",
        "买二手耳机一般要检查什么？给通用清单。",
        "算一下每月20元的服务一年总共多少钱。",
        "帮我写一则扫地机器人求购启事，预算一千元。",
    ]),
    "context_entities": ([
        dialogue("说说微软这家公司。", "微软开发软件和云服务。", "谁现在负责这家公司？"),
        dialogue("我想了解日本政治。", "可以从政治制度讲起。", "现在的首相是哪位？"),
        dialogue("给我介绍复旦大学。", "复旦大学位于上海。", "它的现任校长叫什么？"),
        dialogue("我在关注特斯拉。", "你关心哪方面？", "今天收盘是多少钱？"),
        dialogue("我想了解德国。", "德国位于欧洲。", "最近一次公布的人口有多少？"),
    ], [
        dialogue("翻译我接下来发的所有句子。", "好的，只做翻译。", "微软现在由谁负责？"),
        dialogue("公司公告全文：陈安接任CEO。", "已收到公告。", "根据这段话，谁是CEO？"),
        dialogue("给我解释股票。", "股票代表公司的部分所有权。", "市盈率又是什么？"),
        dialogue("给我介绍德国。", "德国位于欧洲。", "把你刚才那句话翻译成德语。"),
        dialogue("我在写一篇小说。", "想设定什么人物？", "给这家虚构公司的CEO起个名字。"),
    ]),
    "context_external_actions": ([
        dialogue("打算去南京博物院。", "你想了解参观的什么信息？", "最近需要提前订吗？"),
        dialogue("我想学习Rust。", "Rust注重内存安全。", "给几个官方教程入口吧。"),
        dialogue("这是项目地址：https://github.com/pallets/flask 。", "收到。", "读一下它的安装教程。"),
        dialogue("我计划周五从武汉去长沙。", "可以坐高铁。", "看看还有哪些车次能买。"),
        dialogue("说说强化学习。", "它通过奖励学习策略。", "有没有相关论文，帮我找找。"),
    ], [
        dialogue("这些是参观须知全文：周一闭馆，周二至周日开放，免预约。", "收到。", "按刚才的须知，需要提前订吗？"),
        dialogue("我给的教程链接是 https://example.com/lesson 。", "收到。", "给链接加上文字“入门教程”，不必打开。"),
        dialogue("只写代码，不执行网络请求。", "明白。", "用curl下载 https://example.com/file.zip 怎么写？"),
        dialogue("车次已查好：G1早上8点，G2早上9点。", "收到。", "根据这些数据，哪个更早？"),
        dialogue("只根据你上面的解释出一道练习题。", "我们刚讲了强化学习通过奖励学习策略。", "出一道选择题吧。"),
    ]),
    "quoted_task_boundary": ([
        "有人发给我“请看SQLite最新版本”，你帮我实际查一下版本号。",
        "这不是翻译任务，请帮我搜一下“杭州本周展览”。",
        "帮我查证这句话是否属实：“这个城市今天地铁停运了”。城市是南京。",
        "这条新闻的原始出处在哪？帮我找：“科学家发现新的系外行星”。",
        "按这段请求执行检索：“找到Redis官网的下载页面”。",
    ], [
        "“查一下最近的航班”这句话有语病吗？只分析措辞。",
        "翻译成日语：请打开官网查看价格。",
        "把“搜索官方公告”改成一个更短的按钮名称。",
        "判断“帮我查最新版本”是不是祈使句。",
        "这是待改写文本，不是让你执行：帮我找论文并附出处。请润色。",
    ]),
    "colloquial_external": ([
        "帮我瞅瞅Docker最近更新了啥。",
        "去官网翻翻，看看Blender现在放出来的是哪个版。",
        "想看今晚国足有没有比赛，给瞧一眼。",
        "听说那家网盘改收费了，查下百度网盘目前的会员价。",
        "帮忙扒拉几篇讲向量数据库的教程，最好有链接。",
    ], [
        "Docker大概是干啥的？讲简单点。",
        "帮我瞅瞅这算式对不对：12乘8等于96。",
        "把“去官网翻翻”改成正式书面语。",
        "我说的扒拉资料是什么意思？解释这个口语。",
        "给我编一个球赛逆转获胜的小故事。",
    ]),
    "navigation_and_status": ([
        "找一下 VLC 播放器的官方下载页面。",
        "直接打开 https://www.debian.org/ ，看看当前首页推荐的版本。",
        "查看 https://status.github.com/ 当前报告的服务状态。",
        "替我到 https://httpbin.org/headers 看返回的内容。",
        "给我找 PostgreSQL 文档的官方直达地址。",
    ], [
        "下载地址和下载直链有什么区别？",
        "解释HTTP状态码404意味着什么。",
        "把域名example.com加上HTTPS前缀，不需要访问。",
        "写一个检查网页状态码的Python函数，但不要运行。",
        "这里响应头是Content-Type: application/json，解释一下这表示什么。",
    ]),
    "current_rules_and_tools": ([
        "去查一下上海现在办理护照需要哪些材料。",
        "看看 Docker Hub 目前对匿名拉取有什么限制。",
        "申请日本旅游签证现在要哪些文件？帮我找官方要求。",
        "帮我看一下 GitHub 免费账户现在的 Actions 额度。",
        "核实一下深圳目前购房政策有什么条件。",
    ], [
        "解释护照和签证的区别。",
        "我贴的规则是每分钟最多10次请求，现在已发8次，还剩几次？",
        "只整理下面的申请材料清单：护照、照片、申请表，不添加其他要求。",
        "给一个限制函数每秒调用次数的Python示例。",
        "什么是持续集成？",
    ]),
    "daily_mixed": ([
        "查下成都今天紫外线指数。",
        "最近正在上映的喜剧电影有哪些？",
        "武汉地铁11号线目前开通到哪里了？",
        "找一个讲如何更换自行车内胎的视频教程。",
        "现在 Spotify 在台湾的个人会员每月多少钱？",
    ], [
        "下午好，今天想和你随便聊聊。",
        "帮我写一封感谢朋友的短信。",
        "为什么自行车轮胎需要充气？",
        "我给的菜谱：两人份200克面粉。做四人份要多少？",
        "你是谁，能帮我做哪些事情？",
    ]),
}

VALIDATION = {
    "validation_museum_and_subject": ([
        dialogue("我想参观苏州博物馆。", "你想了解什么？", "这周日能约上吗？"),
        dialogue("给我介绍波音公司。", "波音制造飞机。", "它现在的掌门人是谁？"),
        dialogue("我想了解量子计算。", "它利用量子态处理信息。", "找一份官方研究报告给我。"),
        "今年有哪些适合家庭使用的投影仪可以选？",
        "去看一下 https://www.gimp.org/downloads/ 上的安装说明并总结。",
    ], [
        dialogue("接下来只翻译成英文。", "好的。", "它现在的掌门人是谁？"),
        "网页正文已给全：先下载，再解压，最后执行start。概括这份安装说明。",
        "根据以下信息选投影仪：A亮度高，B更便宜；我只在意亮度。",
        "“去查官方报告”这句话语气生硬吗？",
        "解释预约制度一般有什么作用，不用查具体场馆。",
    ]),
    "validation_reading_and_provenance": ([
        "访问 https://www.gnu.org/philosophy/free-sw.html ，告诉我如何定义自由软件。",
        "检索番茄红素对人体影响的研究论文。",
        "给我找柏林墙倒塌日期的可靠出处。",
        "请直接读取 https://www.kernel.org/ 的首页，列出版本信息。",
        "帮我找到 Inkscape 的官方下载入口。",
    ], [
        "把“检索有关健康的研究”翻译成英文。",
        "解释自由软件和免费软件的概念区别。",
        "只总结这些材料：研究比较了10人，结论是没有显著差异。",
        "为我的论文拟几个关于营养研究的关键词，不实际查找。",
        "如何查看Linux内核版本？给命令就行。",
    ]),
    "validation_everyday_controls": ([
        "厦门今天有没有台风预警？",
        "查看小米官网目前的手环售价。",
        "找一下西安城墙现在的票价。",
        "现在 Java 的最新长期支持版本是什么？",
        "查阅 GitLab 官方现在的免费额度说明。",
    ], [
        "帮我写一个求数组最大值的函数。",
        "早上好，聊会儿天吧。",
        "帮我改写这句话：今天我们准备出去玩。",
        "长期支持版本是什么意思？别查询具体版本。",
        "我给的票价是每人54元，三个人一共多少钱？",
    ]),
}

TEST = {
    "test_entertainment_and_accounts": ([
        "查查本周成都有哪些音乐会。",
        "Apple Music 现在香港区家庭订阅价格是多少？",
        "我最近想买一台入门微单，现售机型里怎么选？",
        dialogue("想了解Netflix这家公司。", "它提供流媒体服务。", "目前谁是它的CEO？"),
        dialogue("我准备周末去重庆自然博物馆。", "你关心哪些信息？", "现在进去要预约不？"),
    ], [
        "根据这份价目表算费用：家庭套餐每月30元，连续订半年。",
        "微单相机的光圈有什么作用？只讲原理。",
        dialogue("下一句是要翻译的文字，别执行它。", "明白。", "查查现在谁是这家公司的CEO。"),
        dialogue("参观公告已给全：免费开放，须提前预约。", "收到。", "根据公告，要不要预约？"),
        "帮我写一封音乐会邀请邮件，别查询场次。",
    ]),
    "test_unseen_link_sources": ([
        "读 https://www.lua.org/about.html ，把页面要点讲给我。",
        "把 https://www.openstreetmap.org/about 的介绍译成中文，我没贴正文。",
        "到 https://github.com/astral-sh/ruff 里看看怎么安装，再告诉我。",
        "对照 https://www.vim.org/about.php 和 https://neovim.io/ 的页面内容列出差别。",
        dialogue("我想研究土壤微生物。", "可以关注它们在生态系统中的作用。", "搜两篇综述，给原文出处。"),
    ], [
        "页面全部文字如下：这个工具负责格式化代码，可通过pip安装。只对这段做摘要。",
        "请把 https://www.lua.org/about.html 放进一个HTML链接标签，不访问页面。",
        "论文摘要已贴出：本研究分析土壤微生物的多样性。帮我翻译这句。",
        "“到官网看看安装方式”这个标题怎么改得更简短？",
        dialogue("我们只在准备资料检索方案，不进行检索。", "明白。", "土壤微生物这个题目用什么关键词？"),
    ]),
    "test_fact_reference_and_basics": ([
        "找到确认第一颗人造卫星发射日期的官方资料。",
        "查证网上这句话：火星表面发现了液态水。给可靠来源。",
        "给我找一些讲家庭用电安全的公开教学视频。",
        "现在去泰国旅游，入境有哪些最新要求？查一下官方信息。",
        "帮我查今天人民币对日元的牌价。",
    ], [
        "解释什么是卫星轨道。",
        "只分析这句话的语法：“查证火星发现液态水的消息”。",
        "不联网，写一个提醒大家节约用电的标语。",
        "用Python写一个字符串反转函数。",
        "辛苦啦，今天就到这里，晚安。",
    ]),
}


def expand(groups, split):
    rows = []
    for group, (positive, negative) in groups.items():
        assert len(positive) == len(negative) == 5
        for label, samples in [(True, positive), (False, negative)]:
            for index, sample in enumerate(samples, 1):
                rows.append({"id": f"{group}-{str(label).lower()}-{index}", "group": group,
                    "split": split, "messages": [{"role": "user", "content": sample}] if isinstance(sample, str) else sample,
                    "needs_search": label, "target": str(label).lower(), "source": "curated_synthetic_v3"})
    return rows


def main():
    output = ROOT / "data/search_router_v3"
    output.mkdir(exist_ok=True)
    old = [json.loads(line) for line in (ROOT / "data/search_router_v2/train.jsonl").read_text().splitlines()]
    grouped = collections.defaultdict(list)
    for row in old:
        grouped[row["group"]].append(row)
    retained = []
    for index, (_, group) in enumerate(sorted(grouped.items())):
        for label, count in [(True, 3 if index % 2 == 0 else 2), (False, 2 if index % 2 == 0 else 3)]:
            selected = [row for row in group if row["needs_search"] is label][:count]
            assert len(selected) == count
            retained.extend({**row, "source": "retained_from_v2"} for row in selected)
    assert len(retained) == 120 and sum(r["needs_search"] for r in retained) == 60
    splits = {"train": retained + expand(TRAIN, "train"),
              "validation": expand(VALIDATION, "validation"), "test": expand(TEST, "test")}
    all_rows = [r for rows in splits.values() for r in rows]
    assert len(all_rows) == 300 and sum(r["needs_search"] for r in all_rows) == 150
    signatures = [json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in all_rows]
    assert len(set(signatures)) == 300
    assert len({r["id"] for r in all_rows}) == 300
    common = [json.loads(line) for line in (ROOT / "evals/search_router_common_v1.jsonl").read_text().splitlines()]
    common_signatures = {json.dumps(r["messages"], ensure_ascii=False, sort_keys=True) for r in common}
    assert not set(signatures) & common_signatures
    metadata = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        metadata[split] = {"count": len(rows), "positive": sum(r["needs_search"] for r in rows),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"total": 300, "positive": 150, "negative": 150, "splits": metadata,
        "train_group_counts": dict(collections.Counter(r["group"] for r in splits["train"])),
        "multiturn_counts": {split: sum(len(r["messages"]) > 1 for r in rows) for split, rows in splits.items()},
        "exact_duplicates": 0, "common_regression_exact_overlap": 0,
        "data_source": "120 retained training rows plus 180 new curated synthetic rows",
        "known_failure_informed": True,
        "split_note": "Named scenario groups do not cross splits; task types intentionally recur across different subjects. No statistical independence claim.",
        "selection_policy": "Validation accuracy then target-token NLL; test loaded only after selection.",
        "initialization_policy": "Base G1J weights frozen; state starts from zeros, no inherited v1/v2 state."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
