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

## finalize

最终 RWKV 只读取逐网页摘要二元组和已经完成指代补全的问题。字段审核、审核状态、records、网页原文与重复对话不会进入该调用。

```text
{source_summary_pairs}

问题：{question}
根据以上材料回答问题，合并重复内容；材料不足就说明。事实后只用 [S1] 格式标注来源。
```

二元组格式：

```json
["S1", {"published_date": "2026-08-26", "summary": "一个 RWKV 对一个网页的总结"}]
```

旧的中间 `reduce` 模型调用已经删除；最终模型直接读取全部有界网页摘要。
