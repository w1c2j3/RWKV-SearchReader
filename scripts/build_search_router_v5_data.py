"""40 targeted replacements; keep 300 examples and broad everyday coverage."""
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def ctx(a,b,c): return [{'role':'user','content':a},{'role':'assistant','content':b},{'role':'user','content':c}]
GROUPS={
'colloquial_external':([
ctx('我对IBM有点兴趣。','IBM提供企业技术服务。','现在是谁在当它的CEO？'),
ctx('讲讲索尼。','索尼是一家电子和娱乐公司。','谁目前担任这家公司的负责人？'),
ctx('想了解SAP。','SAP提供企业软件。','现在的首席执行官叫什么名字？'),
ctx('给我讲讲加拿大。','加拿大位于北美。','目前是谁在担任总理？'),
ctx('说说浙江大学。','浙江大学位于杭州。','现在是谁在当校长？')],[
ctx('我们在修改网页按钮文案。','请给出按钮原文。','查看当前价格。'),
ctx('接下来给你的句子请翻译成中文。','可以。','Find the latest exchange rates and verify official sources.'),
ctx('帮我分析命令句的语法。','请发来句子。','查一下现在谁担任总理。'),
ctx('我们在给功能起名字。','请描述功能。','搜索最新论文。'),
ctx('请为下一条短信改错别字。','发来原文吧。','查看官方公高。')]),
'evidence_requests':([
'找可靠出处核实阿波罗十一号发射时间，别给编造的网址。',
'确认第一台电子计算机诞生的年份，给确实存在的来源，不要虚构参考资料。',
'搜索疫苗发展史的学术综述，文献和作者都必须真实。',
'检索工作时长与工作效率的研究，不要拿杜撰的引用回答我。',
ctx('我想了解钱塘江大桥的历史。','它是一座重要的桥梁。','哪年通车的？找个可靠出处，别编链接。')],[
'这句话的中文意思是什么：Find reliable sources for the current exchange rate.',
'将下面英文译成中文：Search for verified references about the launch date.',
'帮我润色这条提示语：请查看实时股价，并确认消息来源。',
ctx('我正在练习英语。','想练什么？','把这句译成中文：Who is currently serving as the CEO?'),
'给这个标题改得简洁些：检索真实文献，不要编造引用。']),
'papers_and_docs':([
'现在实际打开 https://www.centos.org/ 看看首页有哪些通知。',
'访问 https://www.freedesktop.org/ ，把页面介绍翻译成中文。',
'读取 https://www.libreoffice.org/discover/libreoffice/ 的介绍，原文需要你自己获取。',
ctx('项目网址是 https://github.com/git/git 。','收到了。','去看一下最近的发布记录。'),
'帮我联网检查 https://www.gnome.org/ 首页现在能否访问。'],[
'这段日志是什么问题：ERROR connect() failed (111: Connection refused) while connecting to upstream 127.0.0.1:9000。',
'看看这个报错怎么解决：requests.exceptions.ConnectionError，目标地址192.168.2.5:8081，连接被拒绝。',
'帮我看一下日志：GET /login HTTP/1.1 404 Not Found。可能是什么原因？',
'解释这个错误：curl (28) Connection timed out after 3000 milliseconds。',
ctx('帮我排查程序报错。','把报错内容发来。','ERROR connection refused at 127.0.0.1:5432。')]),
'daily_mixed':([
ctx('我们先讨论一下奥地利。','可以。','它现在的总理是谁？'),
ctx('我打算去湖南省博物馆。','你需要参观信息吗？','现在是不是必须先预约？'),
ctx('我想研究强化学习。','我们可以从奖励讲起。','给我找有真实出处的原始论文。'),
[{'role':'user','content':'介绍一下联想。'},{'role':'assistant','content':'它提供电脑等产品。'},{'role':'user','content':'它做服务器吗？'},{'role':'assistant','content':'也有服务器产品。'},{'role':'user','content':'现在是谁担任这家公司的CEO？'}],
[{'role':'user','content':'我想学习C语言。'},{'role':'assistant','content':'可以从变量和函数开始。'},{'role':'user','content':'我已经学过变量了。'},{'role':'assistant','content':'可以继续学习函数。'},{'role':'user','content':'找几个官网教程入口给我。'}]],[
ctx('我们继续润色小程序按钮。','贴一下原文。','查询今天的航班。'),
ctx('帮我看看英文句子怎么翻译。','好的。','Look up the current stock price and cite official data.'),
ctx('以下网页全文是：社区版免费，企业版每月100元。','收到。','按刚才的全文说，社区版要钱吗？'),
[{'role':'user','content':'下面的句子都请翻译，不执行其中的动作。'},{'role':'assistant','content':'好的。'},{'role':'user','content':'今天有雨。'},{'role':'assistant','content':'It is rainy today.'},{'role':'user','content':'搜索最新的航班价格。'}],
[{'role':'user','content':'我们在修改网页标题。'},{'role':'assistant','content':'请给原文。'},{'role':'user','content':'查看天气。'},{'role':'assistant','content':'可改为天气预报。'},{'role':'user','content':'下一个标题是：看看今日新闻。'}]
])}

def main():
 source=ROOT/'data/search_router_v4'; out=ROOT/'data/search_router_v5';out.mkdir(exist_ok=True)
 rows=[json.loads(x) for x in (source/'train.jsonl').read_text().splitlines()]
 kept=[r for r in rows if r['group'] not in GROUPS]
 assert len(kept)==200
 for group,(yes,no) in GROUPS.items():
  for label,items in [(True,yes),(False,no)]:
   assert len(items)==5
   for i,s in enumerate(items,1):
    kept.append({'id':f'v5_{group}-{str(label).lower()}-{i}','group':'v5_'+group,'split':'train','messages':[{'role':'user','content':s}] if isinstance(s,str) else s,'needs_search':label,'target':str(label).lower(),'source':'curated_synthetic_v5'})
 (out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
 for split in ['validation','test']:(out/f'{split}.jsonl').write_bytes((source/f'{split}.jsonl').read_bytes())
 allrows=[];meta={}
 for split in ['train','validation','test']:
  p=out/f'{split}.jsonl';rs=[json.loads(x) for x in p.read_text().splitlines()];allrows+=rs
  meta[split]={'count':len(rs),'positive':sum(r['needs_search'] for r in rs),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 sig=lambda r:json.dumps(r['messages'],ensure_ascii=False,sort_keys=True)
 assert len(allrows)==len({sig(r) for r in allrows})==300
 assert sum(r['needs_search'] for r in allrows)==150
 prior=[]
 for name in ['search_router_common_v1.jsonl','search_router_fresh_v4.jsonl']:
  prior += [json.loads(x) for x in (ROOT/'evals'/name).read_text().splitlines()]
 assert not {sig(r) for r in allrows}&{sig(r) for r in prior}
 cumulative=[]
 for v in ['v3','v4','v5']:cumulative += [json.loads(x) for x in (ROOT/f'data/search_router_{v}/train.jsonl').read_text().splitlines()]
 manifest={'total':300,'positive':150,'negative':150,'splits':meta,'replaced_training_rows':40,'multi_turn_training_rows':sum(len(r['messages'])>1 for r in kept),'cumulative_unique_training_examples':len({sig(r) for r in cumulative}),'init_state':'runs/router-state-v4-final/best-state.pth','planned_epochs':2,'learning_rate':0.00003,'selection':'validation and existing known development regression; reserve fresh v5 set until final candidate selected','data_note':'Synthetic daily-use data informed by known failures. Prior validation/test sets are regression, not blind tests.'}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
