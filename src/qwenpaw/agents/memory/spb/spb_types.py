# -*- coding: utf-8 -*-
"""Core data types for Structured Profile Bootstrapping (SPB)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SPBField:
    key: str
    display_zh: str
    display_en: str
    field_type: str  # "str" | "enum" | "list"
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SPBDimension:
    name: str
    display_name_zh: str
    display_name_en: str
    fields: list[SPBField]


SPB_SCHEMA: list[SPBDimension] = [
    SPBDimension(
        name="demographics",
        display_name_zh="基本信息",
        display_name_en="Basic Information",
        fields=[
            SPBField(key="name", display_zh="姓名", display_en="Name", field_type="str"),
            SPBField(key="preferred_nickname", display_zh="昵称", display_en="Nickname", field_type="str"),
            SPBField(key="pronouns", display_zh="代词", display_en="Pronouns", field_type="enum",
                     options=["he/him", "she/her", "they/them"]),
            SPBField(key="language_preference", display_zh="语言偏好", display_en="Language", field_type="str"),
        ],
    ),
    SPBDimension(
        name="background",
        display_name_zh="背景",
        display_name_en="Background",
        fields=[
            SPBField(key="occupation", display_zh="职业", display_en="Occupation", field_type="str"),
            SPBField(key="expertise_areas", display_zh="专业领域", display_en="Expertise Areas", field_type="list"),
            SPBField(key="current_projects", display_zh="当前项目", display_en="Current Projects", field_type="list"),
        ],
    ),
    SPBDimension(
        name="communication",
        display_name_zh="沟通偏好",
        display_name_en="Communication",
        fields=[
            SPBField(key="tone", display_zh="语气", display_en="Tone", field_type="enum",
                     options=["casual", "formal", "technical"]),
            SPBField(key="response_length", display_zh="回复长度", display_en="Response Length", field_type="enum",
                     options=["concise", "detailed"]),
            SPBField(key="output_language", display_zh="输出语言", display_en="Output Language", field_type="str"),
        ],
    ),
    SPBDimension(
        name="interests",
        display_name_zh="兴趣",
        display_name_en="Interests",
        fields=[
            SPBField(key="topics", display_zh="话题", display_en="Topics", field_type="list"),
            SPBField(key="hobbies", display_zh="爱好", display_en="Hobbies", field_type="list"),
        ],
    ),
    SPBDimension(
        name="preferences",
        display_name_zh="工作偏好",
        display_name_en="Preferences",
        fields=[
            SPBField(key="work_style", display_zh="工作风格", display_en="Work Style", field_type="enum",
                     options=["structured", "flexible"]),
            SPBField(key="decision_style", display_zh="决策风格", display_en="Decision Style", field_type="enum",
                     options=["analytical", "intuitive"]),
            SPBField(key="tools", display_zh="工具", display_en="Tools", field_type="list"),
        ],
    ),
]


def marker_for(dimension: str, field_key: str, field_type: str) -> str:
    return f"<!-- spb:{dimension}.{field_key}:{field_type} -->"
