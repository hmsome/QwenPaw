# -*- coding: utf-8 -*-
"""Extraction prompt templates for SPB per-dimension extraction.

Each prompt takes conversation text + dimension schema, asks the LLM to
extract only explicitly stated information (no inference), and returns
structured JSON.
"""

from __future__ import annotations

from .spb_types import SPBDimension


def _format_schema(dimension: SPBDimension, lang: str = "en") -> str:
    lines = []
    for f in dimension.fields:
        display = f.display_zh if lang == "zh" else f.display_en
        if f.field_type == "enum":
            lines.append(f'  "{f.key}" ({display}): one of {f.options} or null')
        elif f.field_type == "list":
            lines.append(f'  "{f.key}" ({display}): array of strings, or null')
        else:
            lines.append(f'  "{f.key}" ({display}): string or null')
    return "\n".join(lines)


def build_extraction_prompt(
    messages_text: str,
    dimension: SPBDimension,
    language: str = "en",
) -> str:
    """Build extraction prompt for a single dimension.

    Args:
        messages_text: Recent conversation text.
        dimension: The SPBDimension to extract.
        language: Language code for the prompt.

    Returns:
        Formatted extraction prompt.
    """
    schema_str = _format_schema(dimension, language)
    prompts = _PROMPTS.get(language, _PROMPTS["en"])
    return prompts.format(
        dimension_name=dimension.display_name_en,
        schema=schema_str,
        conversation=messages_text,
    )


_EN_PROMPT = """\
You are a profile extraction assistant. Your task is to extract ONLY explicitly stated user profile information from the conversation below.

Dimension: {dimension_name}

Fields to extract:
{schema}

Rules:
1. Only extract information that the user has EXPLICITLY stated — do NOT infer or guess.
2. If a field is not mentioned, set it to null.
3. For list fields, extract all mentioned items.
4. Return valid JSON only.

Conversation:
{conversation}

Respond with a JSON object. Use the field keys exactly as shown above. Example:
{{"field_key": "value", "another_field": ["item1", "item2"]}}

If NO profile information is found for this dimension, respond with:
{{"extracted": false}}
"""

_ZH_PROMPT = """\
你是一个用户资料提取助手。你的任务是从下面的对话中提取用户**明确提到**的个人资料信息。

维度：{dimension_name}

需要提取的字段：
{schema}

规则：
1. 只提取用户**明确说过**的信息，不要推断或猜测。
2. 如果某个字段没有被提及，设为 null。
3. 对于列表类型字段，提取所有提到的项目。
4. 只返回 JSON 格式。

对话：
{conversation}

请返回 JSON 对象，使用上面显示的字段键名。例如：
{{"field_key": "value", "another_field": ["item1", "item2"]}}

如果该维度没有找到任何资料信息，返回：
{{"extracted": false}}
"""

_RU_PROMPT = """\
Вы помощник по извлечению профиля. Ваша задача — извлечь ТОЛЬКО явно указанную информацию о профиле пользователя из приведённого ниже разговора.

Измерение: {dimension_name}

Поля для извлечения:
{schema}

Правила:
1. Извлекайте только информацию, которую пользователь ЯВНО указал — не угадывайте.
2. Если поле не упомянуто, установите null.
3. Для полей-списков извлеките все упомянутые элементы.
4. Возвращайте только JSON.

Разговор:
{conversation}

Ответьте JSON-объектом. Используйте ключи полей точно как показано выше. Пример:
{{"field_key": "value", "another_field": ["item1", "item2"]}}

Если информация для этого измерения не найдена, ответьте:
{{"extracted": false}}
"""

_ID_PROMPT = _EN_PROMPT  # Indonesian uses English prompts

_PROMPTS = {
    "en": _EN_PROMPT,
    "zh": _ZH_PROMPT,
    "ru": _RU_PROMPT,
    "id": _ID_PROMPT,
    "local": _EN_PROMPT,
    "qa": _EN_PROMPT,
}

# Pre-check prompt for Layer 1 LLM fallback
_PRECHECK_PROMPT_EN = """\
Does the following user message contain any personal profile information (name, occupation, preferences, interests, communication style, etc.)?
Answer only "yes" or "no".

User message: {message}
"""

_PRECHECK_PROMPT_ZH = """\
下面这条用户消息是否包含任何个人资料信息（姓名、职业、偏好、兴趣、沟通风格等）？
只回答 "yes" 或 "no"。

用户消息：{message}
"""


def build_precheck_prompt(message: str, language: str = "en") -> str:
    """Build a lightweight pre-check prompt for Layer 1."""
    if language == "zh":
        return _PRECHECK_PROMPT_ZH.format(message=message)
    return _PRECHECK_PROMPT_EN.format(message=message)
