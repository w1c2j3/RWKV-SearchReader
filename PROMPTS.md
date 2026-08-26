# RWKV 输入格式

所有阶段都使用 Naive NoCoT 边界，不发送 system 内容：

```text
User: {阶段输入}

Assistant: <think></think>
```

## plan

```text
当前日期：{date}
问题：{question}

把问题转换为检索计划。decision 只能是 search 或 clarify；公开信息用 search，只有指代无法确定时用 clarify。scope_mode 只能是 fixed 或 discover；明确对象用 fixed，未点名的公开类别用 discover。保留问题中的字段，最多生成 {max_queries} 个检索词。
只输出 JSON：
{"decision":"","scope_mode":"","goal":"","records":[{"subject":"","question":"","fields":[]}],"queries":[],"clarification":""}
```

## scope_candidate

```text
{page_title}

{page_content}

问题：{question}
提取网页中属于问题所问类别的具体对象名称。版本、主题、活动、角色和道具不是对象。最多 4 个。
只输出 JSON：
{"subjects":[]}
```

## scope_discovery

```text
{candidate_rows_json}

问题：{question}
从候选 subject 中原样选择最多 {max_queries} 个具体对象，为每个对象生成记录和检索词。保留问题中的字段，不新增对象。
只输出 JSON：
{"decision":"search","scope_mode":"fixed","goal":"","records":[{"subject":"","question":"","fields":[]}],"queries":[],"clarification":""}
```

## contextualize

```text
{conversation_json_lines}

当前消息：{message}
结合此前对话，把当前消息改写成可独立理解的问题。只输出改写后的问题。
```

## conversation

```text
{conversation_json_lines}

请求：{request}
根据此前对话完成请求。合并重复内容，不复制来源编号。直接回答。
```

## summarize

每个网页独立调用一次 RWKV。这里不发送标题、URL、发布时间、来源编号或用户问题。

```text
{page_content}

总结以上网页内容。
```

## audit

```text
{source_summary_pairs}

记录：{records_json}
问题：{question}
已检索：{queries}
当前日期：{date}

判断材料是否足以覆盖问题和记录中的字段。足够只输出 ENOUGH；不足输出最多 {max_queries} 行新检索词，每行格式为 SEARCH: 检索词。
```

## evidence_select

```text
{source_summary_pairs}

对象：{subject}
字段：{field}
当前日期：{date}
选择一个直接支持该字段的来源；当前信息选择日期最新的来源。没有材料用 missing，冲突用 conflict。
只输出 JSON：
{"source_id":"","status":"","fact":""}
```

## reduce

逐网页摘要超过 final 的材料预算时，按 chunk 并发归并。每个 chunk 最多包含 4 个输入节点且不超过配置的字符预算；归并后仍过大就继续下一层。

```text
{lineage_summary_pairs}

问题：{question}
合并以上材料中能回答问题的内容，删除重复，保留具体名称、日期、数字和 [S1] 来源编号。只输出合并后的材料。
```

归并节点仍是二元组，左侧是不可拆分的来源谱系，右侧是合并摘要：

```json
[["S1", "S2", "S3", "S4"], "合并后的材料 [S1][S3]"]
```

## finalize

最终 RWKV 只读取漏斗最后一层的摘要二元组和已经完成指代补全的问题。典型 20 份网页摘要会并发归并为约 5 份材料，再进行一次最终回答。字段审核、审核状态、records、网页原文与重复对话不会进入该调用。

```text
{final_lineage_summary_pairs}

问题：{question}
根据以上材料直接回答问题，只回答所问内容，合并重复内容，不推断材料没有的信息；材料不足就说明。事实后只用 [S1] 格式标注来源。
```

final 接收的二元组格式：

```json
[["S1", "S2", "S3", "S4"], "一个 chunk 的合并摘要 [S1][S2]"]
```
