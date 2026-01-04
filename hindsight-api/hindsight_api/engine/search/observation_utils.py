"""
Observation utilities for generating entity observations from facts.

Observations are objective facts synthesized from multiple memory facts
about an entity, without personality influence.
"""

import logging

from pydantic import BaseModel, Field

from ..response_models import MemoryFact

logger = logging.getLogger(__name__)


class Observation(BaseModel):
    """An observation about an entity."""

    observation: str = Field(description="观察结论文本：关于该实体的事实性陈述（用中文，专有名词保留原文）")


class ObservationExtractionResponse(BaseModel):
    """Response containing extracted observations."""

    observations: list[Observation] = Field(default_factory=list, description="关于该实体的观察结论列表")


def format_facts_for_observation_prompt(facts: list[MemoryFact]) -> str:
    """Format facts as text for observation extraction prompt."""
    import json

    if not facts:
        return "[]"
    formatted = []
    for fact in facts:
        fact_obj = {"text": fact.text}

        # Add context if available
        if fact.context:
            fact_obj["context"] = fact.context

        # Add occurred_start if available
        if fact.occurred_start:
            fact_obj["occurred_at"] = fact.occurred_start

        formatted.append(fact_obj)

    return json.dumps(formatted, indent=2)


def build_observation_prompt(
    entity_name: str,
    facts_text: str,
) -> str:
    """Build the observation extraction prompt for the LLM."""
    return f"""请根据以下关于「{entity_name}」的事实，生成一组关键“观察结论”。

关于 {entity_name.upper()} 的事实：
{facts_text}

你的任务：把这些事实综合成清晰、客观、可核对的观察结论（关于 {entity_name} 的事实性陈述）。

指南：
1. 每条 observation 都必须是关于 {entity_name} 的**事实性陈述**
2. 适当把相关事实合并成一条更完整的 observation
3. 必须客观：不要加入观点、价值判断、情绪化措辞、推测或心理揣测
4. 只写“我们知道什么”，不要写“我们猜测什么”
5. 优先覆盖：身份/特征/角色/关系/活动/经历（以事实为准）
6. 使用第三人称叙述（例如：“John …”，不要写“我认为 John …”）
7. 如果事实存在冲突，说明“较新的”或“证据更充分”的版本（不要编造）

专有名词规则（必须遵守）：
- 人名、组织名、产品名、项目名、地名等专有名词保持原文，不要翻译或音译（例如：John、Google、Redis、Zoom）。

好的 observation 示例：
- “John 在 Google 担任软件工程师。”
- “John 做事细致、流程化，倾向于用清单和步骤推进任务。”
- “John 经常与 Sarah 在 AI 项目上协作。”
- “John 于 2023 年加入公司。”

不好的 observation 示例（避免）：
- “John 看起来是个好人。”（判断/评价）
- “John 应该很喜欢他的工作。”（推测）
- “我觉得 John 很可靠。”（第一人称观点）

请根据事实生成 3-7 条 observation。如果事实很少，可以生成更少条；不要为了凑数量而编造。"""


def get_observation_system_message() -> str:
    """Get the system message for observation extraction."""
    return "你是一名客观观察者，负责综合关于某个实体的事实，生成清晰、可核对的观察结论。不要输出观点/评价/推测，不要带人格色彩。请用简体中文输出，保持简洁准确。专有名词（人名/组织名/产品名等）保持原文不翻译。"


async def extract_observations_from_facts(llm_config, entity_name: str, facts: list[MemoryFact]) -> list[str]:
    """
    Extract observations from facts about an entity using LLM.

    Args:
        llm_config: LLM configuration to use
        entity_name: Name of the entity to generate observations about
        facts: List of facts mentioning the entity

    Returns:
        List of observation strings
    """
    if not facts:
        return []

    facts_text = format_facts_for_observation_prompt(facts)
    prompt = build_observation_prompt(entity_name, facts_text)

    try:
        result = await llm_config.call(
            messages=[
                {"role": "system", "content": get_observation_system_message()},
                {"role": "user", "content": prompt},
            ],
            response_format=ObservationExtractionResponse,
            scope="memory_extract_observation",
        )

        observations = [op.observation for op in result.observations]
        return observations

    except Exception as e:
        logger.warning(f"Failed to extract observations for {entity_name}: {str(e)}")
        return []
