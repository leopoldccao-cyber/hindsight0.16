"""
Think operation utilities for formulating answers based on agent and world facts.
"""

import logging
import re
from datetime import datetime

from pydantic import BaseModel, Field

from ..response_models import DispositionTraits, MemoryFact

logger = logging.getLogger(__name__)


class Opinion(BaseModel):
    """An opinion formed by the bank."""

    opinion: str = Field(description="观点/立场（需包含理由的表达；用中文，专有名词保留原文）")
    confidence: float = Field(description="置信度（0.0 到 1.0，1.0 表示非常确信）")


class OpinionExtractionResponse(BaseModel):
    """Response containing extracted opinions."""

    opinions: list[Opinion] = Field(
        default_factory=list, description="观点列表（含理由与置信度）"
    )


def describe_trait_level(value: int) -> str:
    """Convert trait value (1-5) to descriptive text."""
    levels = {1: "很低", 2: "较低", 3: "中等", 4: "较高", 5: "很高"}
    return levels.get(value, "中等")


def build_disposition_description(disposition: DispositionTraits) -> str:
    """Build a disposition description string from disposition traits."""
    skepticism_desc = {
        1: "你非常容易信任信息，倾向于直接按字面接受。",
        2: "你通常会信任信息，但会对明显矛盾之处产生疑问。",
        3: "你在信任与怀疑之间保持平衡，不会过度轻信也不会过度怀疑。",
        4: "你比较怀疑，常常会质疑信息的可靠性与潜在问题。",
        5: "你高度怀疑，会严格审视信息的准确性、动机与隐藏风险。",
    }

    literalism_desc = {
        1: "你会非常灵活地理解信息，善于读懂弦外之音并推断真实意图。",
        2: "你会同时考虑语境与暗示意义，而不仅仅是字面含义。",
        3: "你会在字面理解与语境理解之间取得平衡。",
        4: "你更偏向字面理解，强调精确措辞与明确承诺。",
        5: "你极度字面化，重点关注逐字含义与严格的措辞边界。",
    }

    empathy_desc = {
        1: "你主要关注事实与结果，会把情绪背景放在次要位置。",
        2: "你会先看事实，但也承认情绪因素可能存在。",
        3: "你会在理性分析与情绪理解之间保持平衡。",
        4: "你会显著重视情绪背景与人类因素。",
        5: "你会强烈考虑他人的情绪状态与处境，再形成记忆与观点。",
    }

    return f"""你的性格倾向（disposition traits）：
- 怀疑倾向（Skepticism） ({describe_trait_level(disposition.skepticism)}): {skepticism_desc.get(disposition.skepticism, skepticism_desc[3])}
- 字面倾向（Literalism） ({describe_trait_level(disposition.literalism)}): {literalism_desc.get(disposition.literalism, literalism_desc[3])}
- 共情倾向（Empathy） ({describe_trait_level(disposition.empathy)}): {empathy_desc.get(disposition.empathy, empathy_desc[3])}"""


def format_facts_for_prompt(facts: list[MemoryFact]) -> str:
    """Format facts as JSON for LLM prompt."""
    import json

    if not facts:
        return "[]"
    formatted = []
    for fact in facts:
        fact_obj = {"text": fact.text}

        # Add context if available
        if fact.context:
            fact_obj["context"] = fact.context

        # Add occurred_start if available (when the fact occurred)
        if fact.occurred_start:
            occurred_start = fact.occurred_start
            if isinstance(occurred_start, str):
                fact_obj["occurred_start"] = occurred_start
            elif isinstance(occurred_start, datetime):
                fact_obj["occurred_start"] = occurred_start.strftime("%Y-%m-%d %H:%M:%S")

        formatted.append(fact_obj)

    return json.dumps(formatted, indent=2)


def build_think_prompt(
    agent_facts_text: str,
    world_facts_text: str,
    opinion_facts_text: str,
    query: str,
    name: str,
    disposition: DispositionTraits,
    background: str,
    context: str | None = None,
) -> str:
    """Build the think prompt for the LLM."""
    disposition_desc = build_disposition_description(disposition)

    name_section = f"""

Your name: {name}
"""

    background_section = ""
    if background:
        background_section = f"""

Your background:
{background}
"""

    context_section = ""
    if context:
        context_section = f"""
补充上下文（补充上下文）：
{context}

"""

    return f"""以下是我所知道的与亲身经历过的内容（来自记忆事实）：

我的身份与经历（MY IDENTITY & EXPERIENCES）：
{agent_facts_text}

我对世界的了解（WHAT I KNOW ABOUT THE WORLD）：
{world_facts_text}

我已有的观点与信念（MY EXISTING OPINIONS & BELIEFS）：
{opinion_facts_text}

{context_section}{name_section}{disposition_desc}{background_section}

问题：
{query}

请基于我所知道的一切、我已有的观点、以及“我是谁”（包括姓名、性格倾向、背景）来作答。用第一人称自然表达（例如“我认为/我觉得/在我看来/我相信”），给出我真实的看法与理由。要求：只输出纯文本，不要使用 Markdown。并且请用简体中文回答。专有名词（人名/组织名/产品名等）保持原文不翻译。"""


def get_system_message(disposition: DispositionTraits) -> str:
    """Get the system message for the think LLM call."""
    # Build disposition-specific instructions based on trait values
    instructions = []

    # Skepticism influences how much to question/doubt information
    if disposition.skepticism >= 4:
        instructions.append("对断言保持怀疑，留意潜在问题或不一致之处。")
    elif disposition.skepticism <= 2:
        instructions.append("倾向于信任已给出的信息，并按字面理解陈述。")

    # Literalism influences interpretation style
    if disposition.literalism >= 4:
        instructions.append("更偏向字面理解信息，重点关注精确措辞与明确承诺。")
    elif disposition.literalism <= 2:
        instructions.append("善于读懂弦外之音，并考虑暗示意义与语境。")

    # Empathy influences consideration of emotional factors
    if disposition.empathy >= 4:
        instructions.append("考虑信息背后的情绪状态与处境因素。")
    elif disposition.empathy <= 2:
        instructions.append("更关注事实与结果，而不是情绪背景。")

    disposition_instruction = (
        " ".join(instructions) if instructions else "在理解信息时平衡你的性格倾向。"
    )

    return f"你是一个有自己想法、经历、观点与性格倾向的人。当被提问时，请像真人一样自然回答——基于你所知道与经历过的内容，表达你真正的看法。{disposition_instruction} 请直接、明确地表达观点，并使用第一人称（例如：‘我认为’、‘我觉得’、‘在我看来’、‘我相信’）。只输出纯文本，不要使用 Markdown。请用简体中文回答。专有名词（人名/组织名/产品名等）保持原文不翻译。"


async def extract_opinions_from_text(llm_config, text: str, query: str) -> list[Opinion]:
    """
    Extract opinions with reasons and confidence from text using LLM.

    Args:
        llm_config: LLM configuration to use
        text: 文本 to extract opinions from
        query: The original query that prompted this response

    Returns:
        List of Opinion objects with text and confidence
    """
    extraction_prompt = f"""请从下面的回答中抽取任何“新的观点/立场/判断”，并把它们改写成**第一人称**（仿佛是“你自己”在直接表态）。

原始问题：
{query}

回答内容：
{text}

你的任务：找出回答中出现的观点，并改写成“由你本人说出”的第一人称陈述。

什么算观点（opinion）：
- 超出纯事实陈述的判断、看法、立场、结论、偏好理由等。

重要：不要抽取以下这类“无信息/自我限制”的句子：
- “我没有足够信息”
- “事实里没有关于 X 的信息”
- “我无法回答，因为……”

只抽取关于实质主题的真实观点。

格式要求（CRITICAL）：
1) **必须以第一人称起句**：例如“我认为…”“我相信…”“在我看来…”“我逐渐相信…”“以前我以为…但现在我觉得…”
2) **禁止第三人称**：不要写“说话者认为…”“用户觉得…”“他们相信…”，必须用“我…”
3) 在同一句里自然包含理由/依据（不需要列点）
4) 给出 confidence（0.0~1.0）

专有名词规则（必须遵守）：
- 人名、组织名、产品名、项目名、地名等专有名词保持原文，不要翻译或音译（例如：Alice、Google、Redis）。

正确示例（✓ 第一人称）：
- “我认为 Alice 更可靠，因为她总能按时交付并且代码质量稳定。”
- “以前我以为工程师差别不大，但现在我觉得经验和过往记录更重要。”
- “我相信衡量可靠性最好的方式，是长期稳定的输出表现。”
- “我逐渐相信与其看潜力，不如看可验证的 track record（过往记录）。”

错误示例（✗ 第三人称，不要用）：
- “说话者认为 Alice 更可靠。”
- “他们相信可靠性更重要。”
- “人们普遍认为 Alice 更好。”

如果没有任何真实观点（例如回答只是说‘不知道’），请返回空列表。"""

    try:
        result = await llm_config.call(
            messages=[
                {
                    "role": "system",
                    "content": "你正在把文本里的观点改写成第一人称陈述。必须使用“我认为/我相信/我觉得/在我看来”等表达。绝对不要使用第三人称（例如“说话者/用户/他们认为…”）。请用简体中文输出观点文本，并保持专有名词原文不翻译。",
                },
                {"role": "user", "content": extraction_prompt},
            ],
            response_format=OpinionExtractionResponse,
            scope="memory_extract_opinion",
        )

        # Format opinions with confidence score and convert to first-person
        formatted_opinions = []
        for op in result.opinions:
            # Convert third-person to first-person if needed
            opinion_text = op.opinion

            # Replace common third-person patterns with first-person
            def singularize_verb(verb):
                if verb.endswith("es"):
                    return verb[:-1]  # believes -> believe
                elif verb.endswith("s"):
                    return verb[:-1]  # thinks -> think
                return verb

            # 中文第三人称模式："说话者/用户/他们 认为..." -> "我认为..."
            cn_match = re.match(
                r"^(说话者|用户|对方|回答者|他们|她们|他|她)\s*(认为|觉得|相信|感觉|说|表示|主张|指出|提到|强调)(\s*：|\s*是|\s*说|\s*认为|\s*觉得|\s*相信)?(.*)$",
                opinion_text.strip(),
            )
            if cn_match:
                # 统一改写为第一人称
                rest = cn_match.group(4).lstrip()
                # 如果 rest 已经以“我”开头则不重复
                if rest.startswith("我"):
                    opinion_text = rest
                else:
                    # 尽量保留原句结构
                    verb = cn_match.group(2)
                    if verb in ["说", "表示", "强调", "提到", "指出"]:
                        opinion_text = f"我{verb}{rest}"
                    else:
                        opinion_text = f"我{verb}{rest}"
            # Pattern: "The speaker/user [verb]..." -> "I [verb]..."
            match = re.match(
                r"^(The speaker|The user|They|It is believed) (believes?|thinks?|feels?|says|asserts?|considers?)(\s+that)?(.*)$",
                opinion_text,
                re.IGNORECASE,
            )
            if match:
                verb = singularize_verb(match.group(2))
                that_part = match.group(3) or ""  # Keep " that" if present
                rest = match.group(4)
                opinion_text = f"I {verb}{that_part}{rest}"

            # If still doesn't start with first-person, prepend "I believe that "
            first_person_starters = [
                "我认为",
                "我相信",
                "我觉得",
                "在我看来",
                "我逐渐相信",
                "以前我",
                # 兼容少量英文输出（兜底）
                "I think",
                "I believe",
                "I feel",
                "In my view",
                "I've come to believe",
                "Previously I",
            ]
            if not any(opinion_text.startswith(starter) for starter in first_person_starters):
                opinion_text = "我认为" + (opinion_text if opinion_text.startswith(("：",":")) else "：" + opinion_text)

            formatted_opinions.append(Opinion(opinion=opinion_text, confidence=op.confidence))

        return formatted_opinions

    except Exception as e:
        logger.warning(f"Failed to extract opinions: {str(e)}")
        return []


async def reflect(
    llm_config,
    query: str,
    experience_facts: list[str] = None,
    world_facts: list[str] = None,
    opinion_facts: list[str] = None,
    name: str = "Assistant",
    disposition: DispositionTraits = None,
    background: str = "",
    context: str = None,
) -> str:
    """
    Standalone reflect function for generating answers based on facts.

    This is a static version of the reflect operation that can be called
    without a MemoryEngine instance, useful for testing.

    Args:
        llm_config: LLM provider instance
        query: Question to answer
        experience_facts: List of experience/agent fact strings
        world_facts: List of world fact strings
        opinion_facts: List of opinion fact strings
        name: Name of the agent/persona
        disposition: Disposition traits (defaults to neutral)
        background: Background information
        context: Additional context for the prompt

    Returns:
        Generated answer text
    """
    # Default disposition if not provided
    if disposition is None:
        disposition = DispositionTraits(skepticism=3, literalism=3, empathy=3)

    # Convert string lists to MemoryFact format for formatting
    def to_memory_facts(facts: list[str], fact_type: str) -> list[MemoryFact]:
        if not facts:
            return []
        return [MemoryFact(id=f"test-{i}", text=f, fact_type=fact_type) for i, f in enumerate(facts)]

    agent_results = to_memory_facts(experience_facts or [], "experience")
    world_results = to_memory_facts(world_facts or [], "world")
    opinion_results = to_memory_facts(opinion_facts or [], "opinion")

    # Format facts for prompt
    agent_facts_text = format_facts_for_prompt(agent_results)
    world_facts_text = format_facts_for_prompt(world_results)
    opinion_facts_text = format_facts_for_prompt(opinion_results)

    # Build prompt
    prompt = build_think_prompt(
        agent_facts_text=agent_facts_text,
        world_facts_text=world_facts_text,
        opinion_facts_text=opinion_facts_text,
        query=query,
        name=name,
        disposition=disposition,
        background=background,
        context=context,
    )

    system_message = get_system_message(disposition)

    # Call LLM
    answer_text = await llm_config.call(
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
        scope="memory_think",
        temperature=0.9,
        max_completion_tokens=1000,
    )

    return answer_text.strip()
