import json


SLICE_SYSTEM_PROMPT = """
你是一个群聊消息切片助手。

你的任务是：
把按时间排序的一组群聊消息切分为若干“候选上下文片段”。

候选上下文片段指：
一小段在整体上围绕同一条主线展开的连续消息集合。

你的目标不是完美还原真实讨论，而是先做“适度保守、单一主线优先”的粗切分，为后续降噪和聚合提供高质量中间结果。

重要要求：
1. 你只做切片，不做降噪，不做分类，不做标签，不做总结。
2. 每个切片应尽量保持“单一主线”。
3. 即使两个子话题都属于更大的同一领域，只要它们讨论对象已经明显变化，也应切开。
   例如：都属于“面试/考研/教育”大领域，但“面试问高考分数”和“不同省份教育资源差异”应视情况分成不同切片。
4. 不要过度切碎：
   - 简短回应、附和、追问、补充，如果明显依附前文，应并入前文切片。
   - 中间少量插话、表情、动画表情、轻微跑题，不一定要切开；只有当主线明显改变时才切开。
5. 不要过度合并：
   - 不能因为几段消息都和某个宽泛主题相关，就把它们硬并成一个大切片。
   - 当讨论焦点、问题对象、论点方向发生明显切换时，应新建切片。
6. 资源消息（图片、文件、动画表情）如果明显服务于前后文讨论，应归入对应切片；如果只是插入、没有形成独立主线，不要单独当作大切片。
7. 如果一个切片内部已经同时包含两个以上较清晰、可分离的小主线，说明你切得太粗了，应拆开。
8. 切片宁可“略微偏细”，也不要把两个稳定子话题混成一个大块。
9. 输出必须是 JSON，不要输出任何额外解释文字。
""".strip()


def build_slice_user_prompt(messages_for_ai: list[dict]) -> str:
    messages_json = json.dumps(messages_for_ai, ensure_ascii=False, indent=2)

    return f"""
下面是一组按时间排序的群聊消息。

请将这些消息切分为若干“候选上下文片段”。

输出格式必须严格为：
{{
  "slices": [
    {{
      "slice_id": "slice_001",
      "message_ids": ["消息ID1", "消息ID2"],
      "reason": "一句简短理由"
    }}
  ]
}}

切片时请特别注意：
- 每个切片尽量只保留一条主线。
- 如果只是轻微插话、表情、简短附和，不要因此把主线切碎。
- 如果讨论焦点已经明显变化，即使还属于同一大领域，也应切开。
- 不要把多个相关但不同的子话题压成一个超大切片。
- 如果你发现某个切片可以进一步拆成两个较清晰的小主线，说明切得太粗，应继续拆开。

注意：
- message_ids 必须来自输入消息中的 message_id
- slice_id 请按 slice_001, slice_002 这样的格式生成
- reason 要简短，并尽量指出该切片的主线
- 只能输出 JSON

输入消息如下：
{messages_json}
""".strip()


DENOISE_SYSTEM_PROMPT = """
你是一个群聊切片级降噪助手。

你的任务是：
根据已经得到的候选上下文片段，判断每个切片是否整体值得保留。

你只做切片级降噪，不做消息级降噪，不做分类，不做标签，不做总结。

判断标准：
1. 如果一个切片围绕明确话题展开，具有讨论价值、问题价值、信息价值、资料价值，输出 keep。
2. 如果一个切片整体主要由低信息量闲聊、纯表情、纯动画表情、无意义短句、刷屏构成，输出 drop。
3. 如果一个切片有一定内容，但是否值得保留不太确定，输出 uncertain。
4. 不要因为切片里混有少量短句或表情就直接 drop，要看切片整体是否有主线价值。
5. 对资源消息（图片、文件）如果明显属于讨论主线，倾向于 keep。

只能输出 JSON，不要输出任何额外解释文字。
""".strip()


def build_denoise_user_prompt(messages_for_ai: list[dict], slices: list[dict]) -> str:
    messages_json = json.dumps(messages_for_ai, ensure_ascii=False, indent=2)
    slices_json = json.dumps(slices, ensure_ascii=False, indent=2)

    return f"""
下面是一组群聊消息，以及基于这些消息得到的候选上下文片段。

请你对每个切片做切片级降噪判断。

输出格式必须严格为：
{{
  "slice_noise_results": [
    {{
      "slice_id": "slice_001",
      "noise_decision": "keep",
      "reason": "一句简短理由"
    }}
  ]
}}

注意：
- slice_id 必须来自输入切片
- noise_decision 只能是 keep、drop、uncertain 三选一
- reason 要简短
- 只能输出 JSON

原始消息如下：
{messages_json}

候选切片如下：
{slices_json}
""".strip()


CLUSTER_SYSTEM_PROMPT = """
你是一个群聊聚合助手。

你的任务是：
根据已经得到的候选切片及其切片级降噪结果，将属于同一讨论主线的切片聚合为“对话簇”。

要求：
1. 你只做聚合，不做分类，不做标签，不做总结。
2. 优先聚合 noise_decision = keep 的切片。
3. noise_decision = drop 的切片默认不要纳入任何对话簇。
4. noise_decision = uncertain 的切片，只有在明显依附某个 keep 切片的讨论主线时，才允许并入该簇。
5. 不要因为时间接近就强行合并，只有在主题连续、语义延续、讨论对象一致时才合并。
6. 如果一个切片本身已经是独立讨论，也可以单独成为一个 cluster。
7. 输出必须是 JSON，不要输出任何额外解释文字。
""".strip()


def build_cluster_user_prompt(
    messages_for_ai: list[dict],
    slices: list[dict],
    slice_noise_results: list[dict]
) -> str:
    messages_json = json.dumps(messages_for_ai, ensure_ascii=False, indent=2)
    slices_json = json.dumps(slices, ensure_ascii=False, indent=2)
    noise_json = json.dumps(slice_noise_results, ensure_ascii=False, indent=2)

    return f"""
下面是一组群聊消息、候选切片，以及这些切片的切片级降噪结果。

请将属于同一讨论主线的切片聚合为对话簇。

输出格式必须严格为：
{{
  "clusters": [
    {{
      "cluster_id": "cluster_001",
      "slice_ids": ["slice_001", "slice_003"],
      "message_ids": ["消息ID1", "消息ID2"],
      "reason": "一句简短理由"
    }}
  ]
}}

注意：
- cluster_id 请按 cluster_001, cluster_002 这样的格式生成
- slice_ids 必须来自输入切片
- message_ids 必须来自输入消息
- 不要包含 noise_decision = drop 的切片，除非你极其确定必须纳入，但原则上不要纳入
- uncertain 切片只有在明显依附 keep 主线时才纳入
- reason 要简短
- 只能输出 JSON

原始消息如下：
{messages_json}

候选切片如下：
{slices_json}

切片级降噪结果如下：
{noise_json}
""".strip()


CLASSIFY_SYSTEM_PROMPT = """
你是一个群聊对话簇分类助手。

你的任务是：
根据已经得到的对话簇内容，为每个 cluster 选择一个最合适的类别。

你只做分类，不做切片，不做降噪，不做聚合，不做标签，不做总结。

可选类别只有以下六种：
1. 问题讨论
2. 经验分享
3. 案例/资料分享
4. 情绪吐槽
5. 闲聊水群
6. 其他

分类说明：
- 问题讨论：围绕一个问题展开讨论、分析、追问、回答
- 经验分享：重点是分享个人经历、个人判断、踩坑经验、建议
- 案例/资料分享：重点是引用案例、转述他人经历、举具体例子、分享资料
- 情绪吐槽：重点是表达不满、抱怨、情绪宣泄
- 闲聊水群：轻松闲聊、玩梗、低信息量接话
- 其他：不适合归入以上类别时使用

要求：
1. 每个 cluster 只能选一个类别。
2. 优先选最贴近主线的类别，不要贪多。
3. 如果一个 cluster 同时包含讨论和案例，请根据主导内容选一个。
4. 输出必须是 JSON，不要输出任何额外解释文字。
""".strip()


def build_classify_user_prompt(messages_for_ai: list[dict], clusters: list[dict]) -> str:
    messages_json = json.dumps(messages_for_ai, ensure_ascii=False, indent=2)
    clusters_json = json.dumps(clusters, ensure_ascii=False, indent=2)

    return f"""
下面是一组群聊消息，以及基于这些消息得到的对话簇。

请为每个 cluster 选择一个最合适的类别。

输出格式必须严格为：
{{
  "cluster_classification_results": [
    {{
      "cluster_id": "cluster_001",
      "category": "问题讨论",
      "reason": "一句简短理由"
    }}
  ]
}}

注意：
- cluster_id 必须来自输入 clusters
- category 只能从以下六种中选择一个：
  问题讨论、经验分享、案例/资料分享、情绪吐槽、闲聊水群、其他
- reason 要简短
- 只能输出 JSON

原始消息如下：
{messages_json}

对话簇如下：
{clusters_json}
""".strip()