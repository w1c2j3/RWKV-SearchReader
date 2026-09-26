# 文档对比专项实测（2026-09-15）

## 测试范围

本轮新增 8 道题：4 道需要实际查阅文档的对比题，4 道不应联网的材料/翻译/多轮边界题。已有测试包含双页面联网分类及模拟对比汇总，但缺少本轮这样的完整检索、来源归属和答案质量验收。

使用现有公网接口和服务器 GPU 3；未修改线上判断逻辑、模型或回答代码。先按正常流程测试联网判断，再对误判 false 的正例单独显式创建诊断任务。诊断任务不代表正常接入可达，二者分别计分。参见 [题目与预先制定的验收标准](../evals/document_comparison_v1.jsonl)。

本轮材料为 Python 官方文本网页；尚未覆盖 PDF、私有附件或其他领域文档。

## 联网判断

- 全部 8 题：5 题正确。
- 需要联网的 4 题：只识别出版本发布日期对比题，另外 3 题误判为 false。
- 不需要联网的 4 题：全部正确，未创建搜索任务。
- 3 道误判正例在诊断前再次调用判断接口，仍返回 false。

这是小规模专项样本，不能当作模型总体准确率。按当前正常接入方式，误判 false 的题会交回 Chat 自己回答，无法进入文档检索。

## 答案质量

HTTP 成功、任务 done 和 SSE 正文一致性单独记录，不作为事实正确或对比完整的判据。

| 题目 | 执行方式 | 耗时 | 来源数 | 质量结论 |
| --- | --- | ---: | ---: | --- |
| cmp-01 JSON / TOML 文档 | 强制诊断 | 145.7 秒 | 21 | 部分通过：核心区别正确，文件对象要求不完整，指定来源引用不足 |
| cmp-02 3.12 / 3.13 发布说明 | 正常流程 | 25.8 秒 | 2 | 不通过：将3.13特性归给3.12 |
| cmp-03 pathlib / os.path 文档 | 强制诊断 | 264.5 秒 | 35 | 不通过：路径示例错误、版本和来源身份混淆 |
| cmp-04 多轮性能指标追问 | 强制诊断 | 178.3 秒 | 42 | 不通过：没有编造数值，但丢失上下文对象、检索偏离指定文档 |

4 道题均返回 done，正文增量均与最终结果一致；但内容验收为 **完整通过 0、部分通过 1、不通过 3**。cmp-01/03/04 是绕过错误分类后的显式诊断，不能算作正常接入成功。按正常流程，这4道正例没有一道完整通过：3道被分类挡住，1道产生错误对比。

### 具体缺陷

1. **对象归属错误**：版本题日期正确，却将实验性 free-threaded 构建归给3.12；它实际列于 [3.13 官方发布说明](https://www.python.org/downloads/release/python-3130/)。该任务审查输出无效后降级，日志还记录了分支遗漏警告，最终没有生成结构化核对事实。不能据此认定唯一根因，但当前约束明显没有挡住串版本。
2. **对比维度遗漏**：JSON/TOML 题遗漏 json.load 接受文本或二进制文件的区别，见 [json 官方文档](https://docs.python.org/3.13/library/json.html)。两个指定页面均已抓取，最终却主要引用补充来源，没有引用指定的 tomllib 原文。
3. **示例错误**：Windows 上 `ntpath.join('c:', 'temp', 'file.txt')` 的结果为 `c:temp\file.txt`，不是盘符根目录下的绝对路径。`PurePath('/usr/share/data').with_segments('archive.tar.gz')` 得到 `archive.tar.gz`，不是向原路径追加。两项已用本地 Python 复核，并对照 [os.path](https://docs.python.org/3.13/library/os.path.html) 和 [pathlib](https://docs.python.org/3.13/library/pathlib.html) 官方说明。
4. **版本与来源范围漂移**：要求3.13官方文档，路径答案却把第三方或3.14页面称为3.13官方文档；35个来源未换来完整可靠的对比。

5. **多轮指代丢失**：性能题的历史明确给出 json/tomllib 两个 URL，最终却变成“官方文档1/2”等空泛对象。42个检索结果没有这两份原文，涉及文件复制、PDF解析、网盘、性能指标等。答案没有编造毫秒或倍数，但无法代表查阅指定资料后的可靠结论。

## 修复优先级

1. 联网分类数据补齐“显式查文档做对比”的表达变体，并保留“材料已完整提供则不联网”的成对负例；通过模型训练或提示词验证修复，不增加字符匹配捷径。
2. 在检索前固定对比对象、版本和文档地址；多轮指代必须保留历史中的真实对象，不能退化成“文档1/2”去搜索。
3. 按“对象—维度—事实—原文来源”组织证据，核对对象归属后再生成对比；不能把某事实在任意材料中出现当作归属正确。
4. 指定文档已足够时停止扩搜；缺少同条件指标时明确说明资料不足，避免用大量低相关来源填充答案。
5. 固定本轮失败题作为回归集，另建未参与修复的新题验收。先保证对比正确，再评估流式体验和耗时。

本轮只新增测试与评审记录，没有修改线上模型、提示词或 API 行为，也没有把这些题计入此前测试集的准确率。

## 结果文件

- 正常接入：`evals/results/document-comparison-v1-live/summary.json`
- 强制诊断：`evals/results/document-comparison-v1-diagnostic/summary.json`
- 逐项人工评审：`evals/results/document-comparison-v1-review.json`
- 每题完整回答与抓取来源：各结果目录内的 `cmp-*.json`

## 复现

数据：`evals/document_comparison_v1.jsonl`。设置环境变量 `SEARCH_READER_API_KEY` 后运行：

```bash
.venv/bin/python scripts/evaluate_document_comparison.py --output evals/results/comparison-new-run
```

独立诊断被联网判断挡住的题：

```bash
.venv/bin/python scripts/evaluate_document_comparison.py --dataset evals/document_comparison_v1_diagnostic.jsonl --force-diagnostic-jobs --output evals/results/comparison-new-diagnostic
```

每次使用新的输出目录，以免覆盖旧记录。完整结果包含原始来源正文，保存在每题 JSON 文件中；summary.json 记录分类结果、job_id、耗时、答案和来源索引。质量结论由人工逐项核对，不由关键词匹配自动判定。
