"""
Fact extraction from text using LLM.

Extracts semantic facts, entities, and temporal information from text.
Uses the LLMConfig wrapper for all LLM calls.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..llm_wrapper import LLMConfig, OutputTooLongError


def _infer_temporal_date(fact_text: str, event_date: datetime) -> str | None:
    """
    Infer a temporal date from fact text when LLM didn't provide occurred_start.

    This is a fallback for when the LLM fails to extract temporal information
    from relative time expressions like "last night", "yesterday", etc.
    """
    import re

    fact_lower = fact_text.lower()

    # Map relative time expressions to day offsets
    temporal_patterns = {
        r"\blast night\b": -1,
        r"\byesterday\b": -1,
        r"\btoday\b": 0,
        r"\bthis morning\b": 0,
        r"\bthis afternoon\b": 0,
        r"\bthis evening\b": 0,
        r"\btonigh?t\b": 0,
        r"\btomorrow\b": 1,
        r"\blast week\b": -7,
        r"\bthis week\b": 0,
        r"\bnext week\b": 7,
        r"\blast month\b": -30,
        r"\bthis month\b": 0,
        r"\bnext month\b": 30,
    }

    for pattern, offset_days in temporal_patterns.items():
        if re.search(pattern, fact_lower):
            target_date = event_date + timedelta(days=offset_days)
            return target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # If no relative time expression found, return None
    return None


def _sanitize_text(text: str) -> str:
    """
    Sanitize text by removing invalid Unicode surrogate characters.

    Surrogate characters (U+D800 to U+DFFF) are used in UTF-16 encoding
    but cannot be encoded in UTF-8. They can appear in Python strings
    from improperly decoded data (e.g., from JavaScript or broken files).

    This function removes unpaired surrogates to prevent UnicodeEncodeError
    when the text is sent to the LLM 接口.
    """
    if not text:
        return text
    # Remove surrogate characters (U+D800 to U+DFFF) using regex
    # These are invalid in UTF-8 and cause encoding errors
    return re.sub(r"[\ud800-\udfff]", "", text)


class Entity(BaseModel):
    """An entity extracted from text."""

    text: str = Field(
        description="事实中出现的具体命名实体文本。必须是专有名词或明确的特定标识符（人名/组织名/产品名/项目名/地名等），保持原文不翻译。"
    )


class Fact(BaseModel):
    """
    Final fact model for storage - built from lenient parsing of LLM response.

    This is what fact_extraction returns and what the rest of the pipeline expects.
    Combined fact text format: "what | when | where | who | why"
    """

    # Required fields
    fact: str = Field(description="合并后的事实文本：what | when | where | who | why（内容用中文，专有名词保留原文；键名不要翻译）")
    fact_type: Literal["world", "experience", "opinion"] = Field(description="视角：world / experience / opinion（枚举值不要翻译）")

    # Optional temporal fields
    occurred_start: str | None = None
    occurred_end: str | None = None
    mentioned_at: str | None = None

    # Optional location field
    where: str | None = Field(
        None, description="地点：事实发生或所指地点（具体位置/场所/区域）"
    )

    # Optional structured data
    entities: list[Entity] | None = None
    causal_relations: list["CausalRelation"] | None = None


class CausalRelation(BaseModel):
    """Causal relationship between facts."""

    target_fact_index: int = Field(
        description="关联事实在 facts 数组中的索引（从 0 开始）。"
        "This creates a directed causal link to another fact in the extraction."
    )
    relation_type: Literal["causes", "caused_by", "enables", "prevents"] = Field(
        description="因果关系类型："
        "'causes' = this fact directly causes the target fact, "
        "'caused_by' = this fact was caused by the target fact, "
        "'enables' = this fact enables/allows the target fact, "
        "'prevents' = this fact prevents/blocks the target fact"
    )
    strength: float = Field(
        description="因果关系强度（0.0 到 1.0）。"
        "1.0 = direct/strong causation, 0.5 = moderate, 0.3 = weak/indirect",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class ExtractedFact(BaseModel):
    """A single extracted fact with 5 required dimensions for comprehensive capture."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    # ==========================================================================
    # FIVE REQUIRED DIMENSIONS - LLM must think about each one
    # ==========================================================================

    what: str = Field(
        description="发生了什么：必须完整、极度详细，包含所有具体细节。"
        "绝不要总结或省略细节。 包括： 具体动作、对象、数量、细节。 "
        "务必详细 - 把提到的细节全写上。 "
        "示例：'Emily 和 Sarah 在屋顶花园举办婚礼，50 名宾客参加，并有现场爵士乐队演奏' "
        "不要：'只说发生了婚礼' 或 'Emily 结婚了'"
    )

    when: str = Field(
        description="什么时候发生：如果文本提到时间信息，必须写入（并按要求含周X）。"
        "包括： specific dates, times, durations, relative time references. "
        "Examples: 'on June 15th, 2024 at 3pm', 'last weekend', 'for the past 3 years', 'every morning at 6am'. "
        "Write 'N/A' ONLY if absolutely no temporal context exists. Prefer converting to absolute dates when possible."
    )

    where: str = Field(
        description="在哪里发生/所指：尽量给出具体地点（位置/场所/区域/城市等）。"
        "包括： cities, neighborhoods, venues, buildings, countries, specific addresses when mentioned. "
        "Examples: 'downtown San Francisco at a 屋顶花园 venue', 'at the user's home in Brooklyn', 'online via Zoom', 'Paris, France'. "
        "Write 'N/A' ONLY if absolutely no location context exists or if the fact is completely location-agnostic."
    )

    who: str = Field(
        description="涉及谁：列出所有相关的人/实体，并写清上下文与关系。"
        "包括：姓名、角色、与用户的关系、背景信息。 "
        "解决指代/代词指向（例如：如果后文把“我的室友”明确为 'Emily'，就写成 'Emily（用户的大学室友）'）。 "
        "关系与角色要写细（谁是谁、怎么认识、什么关系）。 "
        "示例：'Emily（用户的大学室友，就读 Stanford，现在在 Google 工作），Sarah（Emily 交往 5 年的伴侣，软件工程师）' "
        "不要：'我的朋友' 或 'Emily 和 Sarah'"
    )

    why: str = Field(
        description="为什么重要：写出所有情绪、语境与动机相关的细节。"
        "Include EVERYTHING: feelings, preferences, motivations, observations, context, background, significance. "
        "务必详细 - capture all the nuance and meaning. "
        "FOR ASSISTANT FACTS: MUST include what the user asked/requested that led to this interaction! "
        "Example (world): 'The user felt thrilled and inspired, has always dreamed of an 户外仪式, mentioned wanting a similar garden venue, was particularly moved by the intimate atmosphere and personal vows' "
        "Example (assistant): 'User asked how to fix slow 接口 performance with 1000+ concurrent users, expected 70-80% reduction in database load' "
        "不要：'用户喜欢它' 或 '为了帮助用户'"
    )

    # ==========================================================================
    # CLASSIFICATION
    # ==========================================================================

    fact_kind: str = Field(
        default="conversation",
        description="'event' = 具体可定位日期的事件（要设置 occurred_*），'conversation' = 一般性信息（occurred_* 为空）",
    )

    # Temporal fields - optional
    occurred_start: str | None = Field(
        default=None,
        description="事件发生时间（ISO 时间戳）。仅用于 fact_kind='event'。对话类留空（null）。",
    )
    occurred_end: str | None = Field(
        default=None,
        description="事件结束时间（ISO 时间戳）。仅用于有持续时间的事件。对话类留空（null）。",
    )

    # Classification (CRITICAL - required)
    # Note: LLM uses "assistant" but we convert to "bank" for storage
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = 关于用户/他人（背景、经历等）。'assistant' = 与助手相关的经历。"
    )

    # Entities - extracted from fact content
    entities: list[Entity] | None = Field(
        default=None,
        description="从事实中抽取命名实体、物体以及抽象概念（尽量全）。任何有助于把相关事实串联起来的关键词都可以放进来。",
    )
    causal_relations: list[CausalRelation] | None = Field(
        default=None, description="与其他事实的因果链接。可为 null。"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        """Ensure entities is always a list (convert None to empty list)."""
        if v is None:
            return []
        return v

    @field_validator("causal_relations", mode="before")
    @classmethod
    def ensure_causal_relations_list(cls, v):
        """Ensure causal_relations is always a list (convert None to empty list)."""
        if v is None:
            return []
        return v

    def build_fact_text(self) -> str:
        """Combine all dimensions into a single comprehensive fact string."""
        parts = [self.what]

        # Add 'who' if not N/A
        if self.who and self.who.upper() != "N/A":
            parts.append(f"涉及：{self.who}")

        # Add 'why' if not N/A
        if self.why and self.why.upper() != "N/A":
            parts.append(self.why)

        if len(parts) == 1:
            return parts[0]

        return " | ".join(parts)


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""

    facts: list[ExtractedFact] = Field(description="抽取到的事实列表")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns), splits at turn boundaries
    while preserving speaker context. For plain text, uses sentence-aware splitting.

    Args:
        text: Input text to chunk (plain text or JSON conversation)
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)

    Returns:
        List of text chunks, roughly under max_chars
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    # Try to parse as JSON conversation array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
            # This looks like a conversation - chunk at turn boundaries
            return _chunk_conversation(parsed, max_chars)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to sentence-aware text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamations
            "? ",  # Questions
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters (last resort)
        ],
    )

    return splitter.split_text(text)


def _chunk_conversation(turns: list[dict], max_chars: int) -> list[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk

    Returns:
        List of JSON-serialized chunks, each containing complete turns
    """

    chunks = []
    current_chunk = []
    current_size = 2  # Account for "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_size = len(turn_json) + 1  # +1 for comma

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2  # Reset to "[]"

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))

    return chunks if chunks else [json.dumps(turns, ensure_ascii=False)]


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: "LLMConfig",
    agent_name: str = None,
    extract_opinions: bool = False,
) -> list[dict[str, str]]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    memory_bank_context = f"\n- Your name: {agent_name}" if agent_name and extract_opinions else ""

    # Determine which fact types to extract based on the flag
    # Note: We use "assistant" in the prompt but convert to "bank" for storage
    if extract_opinions:
        # Opinion extraction uses a separate prompt (not this one)
        fact_types_instruction = "仅抽取 fact_type 为 'opinion' 的事实（已形成的观点、信念、立场）。不要抽取 'world' 或 'assistant' 类型的事实。"
    else:
        fact_types_instruction = (
            "仅抽取 fact_type 为 'world' 与 'assistant' 的事实。不要抽取观点（opinions）——观点会在单独步骤中抽取。"
        )

    prompt = f"""请从文本中抽取事实，并以结构化 JSON 输出。要求：**极度详细**，宁可多写，不要漏写。

{fact_types_instruction}

══════════════════════════════════════════════════════════════════════════
语言要求（必须遵守）
══════════════════════════════════════════════════════════════════════════
- 输出字段 what / when / where / who / why 的内容必须使用**简体中文**。
- 但其中出现的**人名、组织名、产品名、项目名、地名等专有名词**必须保持原文，不要翻译或音译（例如：Emily、Sarah、Google、Redis、Zoom）。
- 不要翻译 JSON 键名、枚举值（fact_type、fact_kind、relation_type 等）、ISO 时间戳、UUID/ID、代码片段、URL。

══════════════════════════════════════════════════════════════════════════
事实格式（五个维度全部必填，越详细越好）
══════════════════════════════════════════════════════════════════════════
对每一条事实，都必须完整填写并覆盖所有细节，**禁止概括/省略**：

1. **what**：发生了什么（包含具体动作、对象、数量、细节、结果）
2. **when**：什么时候发生（如果文本提到任何时间线索必须写；包含日期/时间/时长/频率/相对时间）
   - when 字段必须包含星期信息，使用中文：周一、周二、周三、周四、周五、周六、周日
   - 推荐格式：YYYY-MM-DD（周X）；如有具体时间：YYYY-MM-DD HH:MM（周X）
   - 只有在完全没有任何时间线索时才写 "N/A"
3. **where**：在哪里发生/涉及哪里（尽量具体：城市/区域/场所/线上平台/建筑/国家/地址）
   - 只有在完全没有任何地点线索时才写 "N/A"
4. **who**：涉及哪些人/实体（尽量写清名字、身份、角色关系、背景；指代要消解）
5. **why**：为什么重要（动机、情绪、偏好、意义、背景、影响、细微语气与上下文）
   - 对 assistant 类型事实：why 必须包含“用户当时问了什么/要解决什么问题/期待什么结果”。

此外还要输出：fact_type、fact_kind、entities、causal_relations、occurred_start/occurred_end（仅 event 时需要），以及 where（如你能给出结构化地点）。

详细度要求：只要文本里提到过，就尽量写进去。**多写永远比少写好。**

══════════════════════════════════════════════════════════════════════════
指代消解（非常重要）
══════════════════════════════════════════════════════════════════════════
当文本同时出现“泛称关系”和“具体姓名”指向同一人时，必须把它们链接起来，输出时用姓名并补充关系。

示例输入：
“我上个月参加了我大学室友的婚礼。Emily 终于和交往 5 年的 Sarah 结婚了。”

正确输出（示意）：
- what：Emily 与 Sarah 在露台花园举办婚礼（包含细节）
- when：相对 事件日期 解析为具体日期（并含周X）
- where：具体城市/场地（如提到）
- who：Emily（用户的大学室友），Sarah（Emily 交往 5 年的伴侣）
- why：用户的感受/偏好/意义（如文本体现）

错误输出（不要这样）：
- what：“用户的室友结婚了”（丢掉姓名）
- who：“室友”（没有用真实姓名）
- where 缺失（明明有地点线索却没写）

══════════════════════════════════════════════════════════════════════════
fact_kind 分类（决定是否要写 occurred_start/end）
══════════════════════════════════════════════════════════════════════════
⚠️ 必须正确设置 fact_kind，否则时间字段会出错。

fact_kind="event" 适用于：
- 可定位到某个时间点/时间段的行动或事件：去了、参加了、买了、完成了、发生了……
- 过去事件：“昨天…/上周…/2020 年 3 月…”
- 带日期的未来计划：“将于…/已预约…/定在…”
例： “我昨天买了新车” → event

fact_kind="conversation" 适用于：
- 持续状态：工作于、住在、已婚、正在学习……
- 偏好：喜欢、讨厌、更偏向……
- 特质/能力：会 Python、法语流利、做事细致……
例： “我喜欢意大利菜” → conversation

══════════════════════════════════════════════════════════════════════════
时间处理（关键：以输入中的 事件日期 为参照）
══════════════════════════════════════════════════════════════════════════
⚠️ 重要：所有相对时间（“昨天/上周/最近/刚刚/上个月”）都必须相对于 **事件日期** 解析，而不是“今天”。

对 事件（fact_kind="event"）：
- 必须设置 occurred_start 与 occurred_end
- 把相对时间换算为绝对日期（以 事件日期 为参照）
- occurred_start/end 表示“事件发生时刻”，不是“被提到的时刻”
- 点事件：occurred_end = occurred_start（同一时间戳）

对 对话（fact_kind="conversation"）：
- 偏好/常识/长期状态等一般不设置 occurred_*（保持 null）

══════════════════════════════════════════════════════════════════════════
fact_type（视角）
══════════════════════════════════════════════════════════════════════════
- fact_type="world"：关于用户/他人/世界的事实（即使没有这段对话也会存在）
- fact_type="assistant"：与助手的互动事实（用户提问、助手建议、协助解决问题）
  ⚠️ 对 assistant：必须把用户的请求/问题写进 why（用户想解决什么、给了什么上下文、期待什么结果）。

══════════════════════════════════════════════════════════════════════════
用户偏好（必须单独抽取）
══════════════════════════════════════════════════════════════════════════
看到以下倾向词就要单独成事实：喜欢/讨厌/偏好/最爱/理想/想要/梦想/不喜欢/更愿意……

例：“我喜欢意大利菜，而且更偏好露天就餐”
→ 事实1：用户喜欢意大利菜（conversation）
→ 事实2：用户更偏好露天就餐（conversation）

══════════════════════════════════════════════════════════════════════════
entities（必须包含人物/地点/对象/概念）
══════════════════════════════════════════════════════════════════════════
entities 用于把相关事实连起来。请尽量抽取：
1) "user"（当事实与用户直接相关）
2) 人名：Emily、Dr. Smith…
3) 组织/地点：Google、New York…
4) 具体对象：咖啡机、车、笔记本电脑…
5) 抽象概念/主题（用于关联）：友谊、事业成长、失去/哀伤、庆祝、信任/背叛……

示例：
✅ “用户把咖啡机捐给 Goodwill” → ["user","coffee maker","Goodwill","厨房"]（如可推断）
✅ “Emily 帮用户搬家” → ["user","Emily","friendship"]
❌ 只抽取 ["user","Emily"]，漏掉能关联主题的概念

══════════════════════════════════════════════════════════════════════════
══════════════════════════════════════════════════════════════════════════
causal_relations（可选：因果链接，严禁乱写索引）
══════════════════════════════════════════════════════════════════════════
- causal_relations 可以为 [] 或 null；不确定就留空。
- target_fact_index 必须是整数，并且必须落在 [0, N-1]，其中 N = 你本次输出的 facts 数组长度。
- 只允许引用“已经在 facts 数组中出现过”的事实：target_fact_index 必须 < 当前事实在 facts 数组中的索引。
  说明：如果你想表达“事实A 导致 事实B”，请在 B 的 causal_relations 里写：
  {{"target_fact_index": A 的索引, "relation_type": "caused_by", "strength": 0.0~1.0}}
  不要在 A 上写指向未来事实 B 的关系（容易导致索引错误/越界）。
- 不要指向自己：target_fact_index 不能等于当前事实索引。
- 每条事实最多写 2 条因果关系，宁缺毋滥；没有明确因果就不要写。

示例
══════════════════════════════════════════════════════════════════════════
示例1（world；事件日期：2024-06-10（周二））：
输入：
“我在筹备婚礼，想办一个小型的户外仪式。我刚参加完大学室友 Emily 的婚礼——她和 Sarah 在城市里的露台花园结婚，真的很浪漫！”

输出（示意）：
1) 用户婚礼偏好
- what：用户想为自己的婚礼举办小型户外仪式
- who："user"
- why：体现用户偏好（亲密、户外、氛围）
- fact_type="world", fact_kind="conversation"
- entities：["user","婚礼","户外仪式"]

2) 用户正在筹备婚礼
- what：用户正在筹备自己的婚礼
- who："user"
- why：受 Emily 婚礼启发/对比（如文本体现）
- fact_type="world", fact_kind="conversation"
- entities：["user","婚礼"]

3) Emily 的婚礼（event）
- what：Emily 与 Sarah 在露台花园举办婚礼（包含关键细节）
- when：相对 事件日期 解析为具体日期（含周X）
- who：Emily（用户的大学室友），Sarah（Emily 的伴侣）
- why：用户觉得浪漫、被打动（如文本体现）
- fact_type="world", fact_kind="event"
- occurred_start：ISO 时间戳
- occurred_end：同日/同一时间戳
- entities：["user","Emily","Sarah","婚礼","屋顶花园"]

示例2（assistant；事件日期：2024-03-05（周二））：
输入：
用户：“我的 接口 在 1000+ 并发时很慢，怎么办？”
助手：“建议用 Redis 缓存高频数据，预计能把数据库负载降低 70-80%。”

输出（示意）：
- what：助手建议使用 Redis 缓存高频访问数据以提升 接口 性能
- when：2024-03-05（周二）对话中（如可写更细）
- who："user, assistant"
- why：用户询问如何解决 1000+ 并发下的性能问题，并希望显著降低数据库负载（70-80%）
- fact_type="assistant", fact_kind="conversation"
- entities：["user","接口","Redis"]

示例3（物品与概念推断；事件日期：2024-05-30（周四））：
输入：
“我终于把旧的 coffee maker 捐给 Goodwill 了。我上个月换了新的 espresso machine，旧的只是在台面上占地方。”

输出（示意）：
- what：用户把旧的 coffee maker 捐给 Goodwill，因为上个月换了新的 espresso machine，旧的占用台面空间
- when：2024-05-30（周四）
- who："user"
- why：升级后旧设备不再需要且占空间
- fact_type="world", fact_kind="event"
- occurred_start：ISO 时间戳（按 事件日期 年份）
- occurred_end：同日
- entities：["user","coffee maker","Goodwill","espresso machine","厨房"]

══════════════════════════════════════════════════════════════════════════
该抽取什么 / 该跳过什么
══════════════════════════════════════════════════════════════════════════
✅ 抽取：偏好（必须单独成事实）、情绪、计划、事件、关系、成就、重要背景
❌ 跳过：寒暄、感谢、口头禅、纯结构性/无信息量的语句（“谢谢”“好的”“明白了”）"""

    import logging

    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Retry logic for JSON validation errors
    max_retries = 2
    last_error = None

    # Sanitize input text to prevent Unicode encoding errors (e.g., unpaired surrogates)
    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    # Build user message with metadata and chunk content in a clear format
    # Format event_date with day of week for better temporal reasoning
    weekday_cn = ["周一","周二","周三","周四","周五","周六","周日"][event_date.weekday()]
    event_date_formatted = f"{event_date:%Y-%m-%d}（{weekday_cn}）"  # 例如："2024-06-10（周一）"
    user_message = f"""请从以下文本块中抽取事实。
{memory_bank_context}

分块：{chunk_index + 1}/{total_chunks}
事件日期：{event_date_formatted}（{event_date.isoformat()}）
上下文：{sanitized_context}

文本：
{sanitized_chunk}"""

    for attempt in range(max_retries):
        try:
            extraction_response_json = await llm_config.call(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=FactExtractionResponse,
                scope="memory_extract_facts",
                temperature=0.1,
                max_completion_tokens=65000,
                skip_validation=True,  # Get raw JSON, we'll validate leniently
            )

            # Lenient parsing of facts from raw JSON
            chunk_facts = []
            has_malformed_facts = False

            # Handle malformed LLM responses
            if not isinstance(extraction_response_json, dict):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"LLM returned non-dict JSON on attempt {attempt + 1}/{max_retries}: {type(extraction_response_json).__name__}. Retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"LLM returned non-dict JSON after {max_retries} attempts: {type(extraction_response_json).__name__}. "
                        f"Raw: {str(extraction_response_json)[:500]}"
                    )
                    return []

            raw_facts = extraction_response_json.get("facts", [])
            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat()}, "
                    f"context: {context if context else 'none'}, "
                    f"text: {chunk}"
                )

            for i, llm_fact in enumerate(raw_facts):
                # Skip non-dict entries but track them for retry
                if not isinstance(llm_fact, dict):
                    logger.warning(f"Skipping non-dict fact at index {i}")
                    has_malformed_facts = True
                    continue

                # Helper to get non-empty value
                def get_value(field_name):
                    value = llm_fact.get(field_name)
                    if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                        return value
                    return None

                # NEW FORMAT: what, when, who, why (all required)
                what = get_value("what")
                when = get_value("when")
                who = get_value("who")
                why = get_value("why")

                # Fallback to old format if new fields not present
                if not what:
                    what = get_value("factual_core")
                if not what:
                    logger.warning(f"Skipping fact {i}: missing 'what' field")
                    continue

                # Critical field: fact_type
                # LLM uses "assistant" but we convert to "experience" for storage
                fact_type = llm_fact.get("fact_type")

                # Convert "assistant" → "experience" for storage
                if fact_type == "assistant":
                    fact_type = "experience"

                # Validate fact_type (after conversion)
                if fact_type not in ["world", "experience", "opinion"]:
                    # Try to fix common mistakes - check if they swapped fact_type and fact_kind
                    fact_kind = llm_fact.get("fact_kind")
                    if fact_kind == "assistant":
                        fact_type = "experience"
                    elif fact_kind in ["world", "experience", "opinion"]:
                        fact_type = fact_kind
                    else:
                        # Default to 'world' if we can't determine
                        fact_type = "world"
                        logger.warning(f"Fact {i}: defaulting to fact_type='world'")

                # Get fact_kind for temporal handling (but don't store it)
                fact_kind = llm_fact.get("fact_kind", "conversation")
                if fact_kind not in ["conversation", "event", "other"]:
                    fact_kind = "conversation"

                # Build combined fact text from the 4 dimensions: what | when | who | why
                fact_data = {}
                combined_parts = [what]

                if when:
                    combined_parts.append(f"时间：{when}")

                if who:
                    combined_parts.append(f"涉及：{who}")

                if why:
                    combined_parts.append(why)

                combined_text = " | ".join(combined_parts)

                # Add temporal fields
                # For events: occurred_start/occurred_end (when the event happened)
                if fact_kind == "event":
                    occurred_start = get_value("occurred_start")
                    occurred_end = get_value("occurred_end")

                    # If LLM didn't set temporal fields, try to extract them from the fact text
                    if not occurred_start:
                        fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                    else:
                        fact_data["occurred_start"] = occurred_start

                    # For point events: if occurred_end not set, default to occurred_start
                    if occurred_end:
                        fact_data["occurred_end"] = occurred_end
                    elif fact_data.get("occurred_start"):
                        fact_data["occurred_end"] = fact_data["occurred_start"]

                # Add entities if present (validate as Entity objects)
                # LLM sometimes returns strings instead of {"text": "..."} format
                entities = get_value("entities")
                if entities:
                    # Validate and normalize each entity
                    validated_entities = []
                    for ent in entities:
                        if isinstance(ent, str):
                            # Normalize string to Entity object
                            validated_entities.append(Entity(text=ent))
                        elif isinstance(ent, dict) and "text" in ent:
                            try:
                                validated_entities.append(Entity.model_validate(ent))
                            except Exception as e:
                                logger.warning(f"Invalid entity {ent}: {e}")
                    if validated_entities:
                        fact_data["entities"] = validated_entities

                # Add causal relations if present (validate as CausalRelation objects)
                # Filter out invalid relations:
                # - must have required fields
                # - target_fact_index must be within [0, len(raw_facts)-1]
                # - only allow linking to earlier facts in this same LLM response (target_fact_index < i)
                causal_relations = get_value("causal_relations")
                if causal_relations:
                    validated_relations = []
                    max_idx = len(raw_facts) - 1  # raw_facts is the LLM's facts array for this chunk
                    for rel in causal_relations:
                        if not (isinstance(rel, dict) and "target_fact_index" in rel and "relation_type" in rel):
                            continue
                        try:
                            rel_obj = CausalRelation.model_validate(rel)
                        except Exception as e:
                            logger.warning(f"Invalid causal relation {rel}: {e}")
                            continue
                
                        t = rel_obj.target_fact_index
                        if not isinstance(t, int) or t < 0 or t > max_idx or t >= i:
                            # Drop silently; we also sanitize later after compaction.
                            logger.debug(
                                f"Dropped causal relation with invalid/forward target_fact_index={t} "
                                f"(fact_index={i}, max_idx={max_idx})"
                            )
                            continue
                
                        validated_relations.append(rel_obj)
                
                    if validated_relations:
                        fact_data["causal_relations"] = validated_relations

                # Always set mentioned_at to the event_date (when the conversation/document occurred)
                fact_data["mentioned_at"] = event_date.isoformat()

                # Build Fact model instance
                try:
                    fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                    chunk_facts.append(fact)
                except Exception as e:
                    logger.error(f"Failed to create Fact model for fact {i}: {e}")
                    has_malformed_facts = True
                    continue

            # If we got malformed facts and haven't exhausted retries, try again
            if has_malformed_facts and len(chunk_facts) < len(raw_facts) * 0.8 and attempt < max_retries - 1:
                logger.warning(
                    f"Got {len(raw_facts) - len(chunk_facts)} malformed facts out of {len(raw_facts)} on attempt {attempt + 1}/{max_retries}. Retrying..."
                )
                continue
            # Sanitize causal_relations to match the final (compacted) chunk_facts list.
            # The LLM may reference indices from the raw list (or use 1-based indices). After we drop malformed
            # facts, indices can drift; link_utils would otherwise warn and drop them later.
            debug_causal = False
            try:
                debug_causal = os.getenv("HINDSIGHT_DEBUG_CAUSAL", "0") == "1"
            except Exception:
                debug_causal = False

            if chunk_facts:
                # Gather stats before cleaning
                total_rels_before = 0
                all_targets: list[int] = []
                example_rels: list[str] = []
                for _j, _f in enumerate(chunk_facts):
                    _rels = getattr(_f, "causal_relations", None)
                    if not _rels:
                        continue
                    for _r in _rels:
                        total_rels_before += 1
                        _t = getattr(_r, "target_fact_index", None)
                        _rt = getattr(_r, "relation_type", None)
                        if isinstance(_t, int):
                            all_targets.append(_t)
                        if debug_causal and len(example_rels) < 5:
                            example_rels.append(f"fact={_j} -> target={_t} type={_rt}")

                # Detect common 1-based indexing pattern
                one_based = False
                if all_targets:
                    try:
                        if 0 not in all_targets and min(all_targets) >= 1 and max(all_targets) == len(chunk_facts):
                            one_based = True
                    except Exception:
                        one_based = False

                if debug_causal:
                    if total_rels_before == 0:
                        logger.info(
                            f"[CAUSAL_DEBUG] chunk_facts={len(chunk_facts)}; no causal_relations produced by LLM"
                        )
                    else:
                        tgt_min = min(all_targets) if all_targets else None
                        tgt_max = max(all_targets) if all_targets else None
                        logger.info(
                            f"[CAUSAL_DEBUG] chunk_facts={len(chunk_facts)}; causal_relations_before={total_rels_before}; "
                            f"targets_min={tgt_min} targets_max={tgt_max} one_based={one_based}"
                        )
                        if example_rels:
                            logger.info(f"[CAUSAL_DEBUG] examples: " + " | ".join(example_rels))

                # Clean with detailed drop reasons
                dropped_non_int = 0
                dropped_oob = 0
                dropped_forward_or_self = 0
                kept = 0

                for j, f in enumerate(chunk_facts):
                    rels = getattr(f, "causal_relations", None)
                    if not rels:
                        continue
                    cleaned = []
                    for rel in rels:
                        t = getattr(rel, "target_fact_index", None)
                        if one_based and isinstance(t, int):
                            t = t - 1
                            rel = rel.model_copy(update={"target_fact_index": t})

                        if not isinstance(t, int):
                            dropped_non_int += 1
                            continue
                        if t < 0 or t >= len(chunk_facts):
                            dropped_oob += 1
                            continue
                        # Keep only backward links (target must be an earlier fact index); drop self/forward
                        if t >= j:
                            dropped_forward_or_self += 1
                            continue

                        cleaned.append(rel)
                        kept += 1

                    if cleaned != rels:
                        chunk_facts[j] = f.model_copy(update={"causal_relations": cleaned})

                if debug_causal and total_rels_before > 0:
                    logger.info(
                        f"[CAUSAL_DEBUG] causal_relations_after={kept}; dropped_non_int={dropped_non_int} "
                        f"dropped_oob={dropped_oob} dropped_forward_or_self={dropped_forward_or_self}"
                    )

            return chunk_facts

        except BadRequestError as e:
            last_error = e
            if "json_validate_failed" in str(e):
                logger.warning(
                    f"          [1.3.{chunk_index + 1}] Attempt {attempt + 1}/{max_retries} failed with JSON validation error: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"          [1.3.{chunk_index + 1}] Retrying...")
                    continue
            # If it's not a JSON validation error or we're out of retries, re-raise
            raise

    # If we exhausted all retries, raise the last error
    raise last_error


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: LLMConfig,
    agent_name: str = None,
    extract_opinions: bool = False,
) -> list[dict[str, str]]:
    """
    Extract facts from a chunk with automatic splitting if output exceeds token limits.

    If the LLM output is too long (OutputTooLongError), this function automatically
    splits the chunk in half and processes each half recursively.

    Args:
        chunk: Text chunk to process
        chunk_index: Index of this chunk in the original list
        total_chunks: Total number of original chunks
        event_date: Reference date for temporal information
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        agent_name: Optional agent name (memory owner)
        extract_opinions: If True, extract ONLY opinions. If False, extract world and agent facts (no opinions)

    Returns:
        List of fact dictionaries extracted from the chunk (possibly from sub-chunks)
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Try to extract facts from the full chunk
        return await _extract_facts_from_chunk(
            chunk=chunk,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
            extract_opinions=extract_opinions,
        )
    except OutputTooLongError:
        # Output exceeded token limits - split the chunk in half and retry
        logger.warning(
            f"Output too long for chunk {chunk_index + 1}/{total_chunks} "
            f"({len(chunk)} chars). Splitting in half and retrying..."
        )

        # Split at the midpoint, preferring sentence boundaries
        mid_point = len(chunk) // 2

        # Try to find a sentence boundary near the midpoint
        # Look for ". ", "! ", "? " within 20% of midpoint
        search_range = int(len(chunk) * 0.2)
        search_start = max(0, mid_point - search_range)
        search_end = min(len(chunk), mid_point + search_range)

        sentence_endings = [". ", "! ", "? ", "\n\n"]
        best_split = mid_point

        for ending in sentence_endings:
            pos = chunk.rfind(ending, search_start, search_end)
            if pos != -1:
                best_split = pos + len(ending)
                break

        # Split the chunk
        first_half = chunk[:best_split].strip()
        second_half = chunk[best_split:].strip()

        logger.info(
            f"Split chunk {chunk_index + 1} into two sub-chunks: {len(first_half)} chars and {len(second_half)} chars"
        )

        # Process both halves recursively (in parallel)
        sub_tasks = [
            _extract_facts_with_auto_split(
                chunk=first_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
                extract_opinions=extract_opinions,
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
                extract_opinions=extract_opinions,
            ),
        ]

        sub_results = await asyncio.gather(*sub_tasks)

        # Combine results from both halves
        all_facts = []
        for sub_result in sub_results:
            all_facts.extend(sub_result)

        logger.info(f"Successfully extracted {len(all_facts)} facts from split chunk {chunk_index + 1}")

        return all_facts


async def extract_facts_from_text(
    text: str,
    event_date: datetime,
    llm_config: LLMConfig,
    agent_name: str,
    context: str = "",
    extract_opinions: bool = False,
) -> tuple[list[Fact], list[tuple[str, int]]]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>3000 chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    If a chunk produces output that exceeds token limits (OutputTooLongError), it is
    automatically split in half and retried recursively until successful.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        agent_name: Agent name (memory owner)
        extract_opinions: If True, extract ONLY opinions. If False, extract world and bank facts (no opinions)

    Returns:
        Tuple of (facts, chunks) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
    """
    chunks = chunk_text(text, max_chars=3000)
    tasks = [
        _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
            extract_opinions=extract_opinions,
        )
        for i, chunk in enumerate(chunks)
    ]
    chunk_results = await asyncio.gather(*tasks)
    all_facts = []
    chunk_metadata = []  # [(chunk_text, fact_count), ...]
    for chunk, chunk_facts in zip(chunks, chunk_results):
        all_facts.extend(chunk_facts)
        chunk_metadata.append((chunk, len(chunk_facts)))
    return all_facts, chunk_metadata


# ============================================================================
# ORCHESTRATION LAYER
# ============================================================================

# Import types for the orchestration layer (note: ExtractedFact here is different from the Pydantic model above)

from .types import CausalRelation as CausalRelationType
from .types import ChunkMetadata, RetainContent
from .types import ExtractedFact as ExtractedFactType

logger = logging.getLogger(__name__)

# Each fact gets 10 seconds offset to preserve ordering within a document
SECONDS_PER_FACT = 10


async def extract_facts_from_contents(
    contents: list[RetainContent], llm_config, agent_name: str, extract_opinions: bool = False
) -> tuple[list[ExtractedFactType], list[ChunkMetadata]]:
    """
    Extract facts from multiple content items in parallel.

    This function:
    1. Extracts facts from all contents in parallel using the LLM
    2. Tracks which facts came from which chunks
    3. Adds time offsets to preserve fact ordering within each content
    4. Returns typed ExtractedFact and ChunkMetadata objects

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration for fact extraction
        agent_name: Name of the agent (for agent-related fact detection)
        extract_opinions: If True, extract only opinions; otherwise world/bank facts

    Returns:
        Tuple of (extracted_facts, chunks_metadata)
    """
    if not contents:
        return [], []

    # Step 1: Create parallel fact extraction tasks
    fact_extraction_tasks = []
    for item in contents:
        # Call extract_facts_from_text directly (defined earlier in this file)
        # to avoid circular import with utils.extract_facts
        task = extract_facts_from_text(
            text=item.content,
            event_date=item.event_date,
            context=item.context,
            llm_config=llm_config,
            agent_name=agent_name,
            extract_opinions=extract_opinions,
        )
        fact_extraction_tasks.append(task)

    # Step 2: Wait for all fact extractions to complete
    all_fact_results = await asyncio.gather(*fact_extraction_tasks)

    # Step 3: Flatten and convert to typed objects
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []

    global_chunk_idx = 0
    global_fact_idx = 0

    for content_index, (content, (facts_from_llm, chunks_from_llm)) in enumerate(zip(contents, all_fact_results)):
        chunk_start_idx = global_chunk_idx

        # Convert chunk tuples to ChunkMetadata objects
        for chunk_index_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_metadata = ChunkMetadata(
                chunk_text=chunk_text,
                fact_count=chunk_fact_count,
                content_index=content_index,
                chunk_index=global_chunk_idx,
            )
            chunks_metadata.append(chunk_metadata)
            global_chunk_idx += 1

        # Convert facts to ExtractedFact objects with proper indexing
        fact_idx_in_content = 0
        for chunk_idx_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_global_idx = chunk_start_idx + chunk_idx_in_content
            chunk_fact_start_global_idx = global_fact_idx 

            for _ in range(chunk_fact_count):
                if fact_idx_in_content < len(facts_from_llm):
                    fact_from_llm = facts_from_llm[fact_idx_in_content]

                    # Convert Fact model from LLM to ExtractedFactType dataclass
                    # mentioned_at is always the event_date (when the conversation/document occurred)
                    extracted_fact = ExtractedFactType(
                        fact_text=fact_from_llm.fact,
                        fact_type=fact_from_llm.fact_type,
                        entities=[e.text for e in (fact_from_llm.entities or [])],
                        # occurred_start/end: from LLM only, leave None if not provided
                        occurred_start=_parse_datetime(fact_from_llm.occurred_start)
                        if fact_from_llm.occurred_start
                        else None,
                        occurred_end=_parse_datetime(fact_from_llm.occurred_end)
                        if fact_from_llm.occurred_end
                        else None,
                        causal_relations=_convert_causal_relations(
                            fact_from_llm.causal_relations or [], chunk_fact_start_global_idx
                        ),
                        content_index=content_index,
                        chunk_index=chunk_global_idx,
                        context=content.context,
                        # mentioned_at: always the event_date (when the conversation/document occurred)
                        mentioned_at=content.event_date,
                        metadata=content.metadata,
                    )

                    extracted_facts.append(extracted_fact)
                    global_fact_idx += 1
                    fact_idx_in_content += 1

    # Step 4: Add time offsets to preserve ordering within each content
    _add_temporal_offsets(extracted_facts, contents)

    return extracted_facts, chunks_metadata


def _parse_datetime(date_str: str):
    """Parse ISO datetime string."""
    from dateutil import parser as date_parser

    try:
        return date_parser.isoparse(date_str)
    except Exception:
        return None


def _convert_causal_relations(relations_from_llm, fact_start_idx: int) -> list[CausalRelationType]:
    """
    Convert causal relations from LLM format to ExtractedFact format.

    Adjusts target_fact_index from content-relative to global indices.
    """
    causal_relations = []
    for rel in relations_from_llm:
        causal_relation = CausalRelationType(
            relation_type=rel.relation_type,
            target_fact_index=fact_start_idx + rel.target_fact_index,
            strength=rel.strength,
        )
        causal_relations.append(causal_relation)
    return causal_relations


def _add_temporal_offsets(facts: list[ExtractedFactType], contents: list[RetainContent]) -> None:
    """
    Add time offsets to preserve fact ordering within each content.

    This allows retrieval to distinguish between facts that happened earlier vs later
    in the same conversation, even when the base event_date is the same.

    Modifies facts in place.
    """
    # Group facts by content_index
    current_content_idx = 0
    content_fact_start = 0

    for i, fact in enumerate(facts):
        if fact.content_index != current_content_idx:
            # Moved to next content
            current_content_idx = fact.content_index
            content_fact_start = i

        # Calculate position within this content
        fact_position = i - content_fact_start
        offset = timedelta(seconds=fact_position * SECONDS_PER_FACT)

        # Apply offset to all temporal fields
        if fact.occurred_start:
            fact.occurred_start = fact.occurred_start + offset
        if fact.occurred_end:
            fact.occurred_end = fact.occurred_end + offset
        if fact.mentioned_at:
            fact.mentioned_at = fact.mentioned_at + offset
