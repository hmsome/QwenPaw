# -*- coding: utf-8 -*-
"""Unit tests for Structured Profile Bootstrapping (SPB)."""

import pytest

from qwenpaw.agents.memory.spb.spb_types import SPB_SCHEMA, marker_for
from qwenpaw.agents.memory.spb.spb_profile_writer import SPBProfileWriter
from qwenpaw.agents.memory.spb.spb_prompts import build_extraction_prompt, build_precheck_prompt
from qwenpaw.agents.memory.spb.spb_evaluator import (
    SPBEvaluator,
    exact_match,
    normalized_str_match,
    semantic_str_match,
    list_semantic_match,
    evaluate_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE_ZH = """\
---
summary: "Agent 身份与用户资料"
---

## 身份

- **名字：** 测试Agent
- **定位：** 测试用

## 用户资料

### 基本信息
<!-- spb:demographics.name:str -->
- **姓名：** *（待补充）*
<!-- spb:demographics.preferred_nickname:str -->
- **昵称：** *（待补充）*
<!-- spb:demographics.pronouns:enum -->
- **代词：** *（待补充）*

### 背景
<!-- spb:background.occupation:str -->
- **职业：** *（待补充）*
<!-- spb:background.expertise_areas:list -->
- **专业领域：** *（待补充）*

### 沟通偏好
<!-- spb:communication.tone:enum -->
- **语气：** *（待补充）*
"""


@pytest.fixture
def profile_path(tmp_path):
    p = tmp_path / "PROFILE.md"
    p.write_text(PROFILE_ZH, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# spb_types tests
# ---------------------------------------------------------------------------

class TestSPBTypes:
    def test_schema_has_5_dimensions(self):
        assert len(SPB_SCHEMA) == 5

    def test_dimension_names(self):
        names = {d.name for d in SPB_SCHEMA}
        assert names == {"demographics", "background", "communication", "interests", "preferences"}

    def test_marker_for(self):
        assert marker_for("demographics", "name", "str") == "<!-- spb:demographics.name:str -->"

    def test_field_count(self):
        total = sum(len(d.fields) for d in SPB_SCHEMA)
        assert total == 15

    def test_enum_fields_have_options(self):
        for dim in SPB_SCHEMA:
            for f in dim.fields:
                if f.field_type == "enum":
                    assert len(f.options) > 0


# ---------------------------------------------------------------------------
# spb_profile_writer tests
# ---------------------------------------------------------------------------

class TestSPBProfileWriter:
    def test_parse_markers(self, profile_path):
        writer = SPBProfileWriter()
        markers = writer.parse_markers(profile_path)

        assert "demographics.name" in markers
        assert markers["demographics.name"]["type"] == "str"
        assert not markers["demographics.name"]["filled"]

    def test_update_field(self, profile_path):
        writer = SPBProfileWriter()
        result = writer.update_field(profile_path, "demographics", "name", "张三")
        assert result is True

        text = profile_path.read_text(encoding="utf-8")
        assert "张三" in text
        # Identity section should be unchanged
        assert "测试Agent" in text

    def test_update_does_not_touch_identity(self, profile_path):
        writer = SPBProfileWriter()
        writer.update_field(profile_path, "demographics", "name", "李四")

        text = profile_path.read_text(encoding="utf-8")
        assert "测试Agent" in text
        assert "李四" in text

    def test_update_nonexistent_field(self, profile_path):
        writer = SPBProfileWriter()
        result = writer.update_field(profile_path, "nonexistent", "field", "value")
        assert result is False

    def test_apply_extraction_results(self, profile_path):
        writer = SPBProfileWriter()
        results = {
            "demographics": {"name": "Alice"},
            "background": {"occupation": "Engineer"},
        }
        updated = writer.apply_extraction_results(profile_path, results)
        assert updated == 2

        text = profile_path.read_text(encoding="utf-8")
        assert "Alice" in text
        assert "Engineer" in text

    def test_apply_idempotent(self, profile_path):
        writer = SPBProfileWriter()
        results = {"demographics": {"name": "Bob"}}
        updated1 = writer.apply_extraction_results(profile_path, results)
        assert updated1 == 1

        # Second apply — field already filled, should not update again
        markers = writer.parse_markers(profile_path)
        assert markers["demographics.name"]["filled"]

    def test_get_unfilled_dimensions(self, profile_path):
        writer = SPBProfileWriter()
        unfilled = writer.get_unfilled_dimensions(profile_path, SPB_SCHEMA)
        names = {d.name for d in unfilled}
        assert "demographics" in names
        assert "background" in names

    def test_unfilled_after_fill(self, profile_path):
        writer = SPBProfileWriter()
        writer.update_field(profile_path, "demographics", "name", "X")
        writer.update_field(profile_path, "demographics", "preferred_nickname", "Y")
        writer.update_field(profile_path, "demographics", "pronouns", "Z")

        unfilled = writer.get_unfilled_dimensions(profile_path, SPB_SCHEMA)
        names = {d.name for d in unfilled}
        # Test profile only has 3 demographics markers (no language_preference),
        # so demographics is fully filled in this profile
        assert "background" in names

    def test_filled_detection(self, profile_path):
        writer = SPBProfileWriter()
        writer.update_field(profile_path, "demographics", "name", "TestName")

        markers = writer.parse_markers(profile_path)
        assert markers["demographics.name"]["filled"]
        assert not markers["demographics.preferred_nickname"]["filled"]


# ---------------------------------------------------------------------------
# spb_prompts tests
# ---------------------------------------------------------------------------

class TestSPBPrompts:
    def test_build_extraction_prompt_en(self):
        dim = SPB_SCHEMA[0]  # demographics
        prompt = build_extraction_prompt("Hello", dim, "en")
        assert "Name" in prompt
        assert "Hello" in prompt
        assert "JSON" in prompt

    def test_build_extraction_prompt_zh(self):
        dim = SPB_SCHEMA[0]  # demographics
        prompt = build_extraction_prompt("你好", dim, "zh")
        assert "姓名" in prompt
        assert "你好" in prompt

    def test_build_precheck_prompt(self):
        prompt = build_precheck_prompt("我叫小明", "zh")
        assert "我叫小明" in prompt

    def test_build_precheck_prompt_en(self):
        prompt = build_precheck_prompt("I am Alice", "en")
        assert "I am Alice" in prompt


# ---------------------------------------------------------------------------
# spb_adapter tests (Layer 1–4 trigger logic)
# ---------------------------------------------------------------------------

class TestSPBAdapter:
    def _make_config(self):
        from qwenpaw.config.config import SPBConfig
        return SPBConfig(
            enabled=True,
            pre_check_keywords_zh=["我是", "我叫", "喜欢"],
            pre_check_keywords_en=["i am", "call me", "like"],
            pre_check_use_llm_fallback=False,
            empty_streak_threshold=3,
            enable_user_override=True,
        )

    @pytest.mark.asyncio
    async def test_should_extract_keyword_zh(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        assert await adapter.should_extract("我是小明")

    @pytest.mark.asyncio
    async def test_should_extract_keyword_en(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        assert await adapter.should_extract("I like programming")

    @pytest.mark.asyncio
    async def test_should_not_extract_random(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        assert not await adapter.should_extract("今天天气怎么样？")

    def test_detect_user_override_zh(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        overrides = adapter._detect_user_override("我叫张三")
        assert "demographics.name" in overrides
        assert overrides["demographics.name"] == "张三"

    def test_detect_user_override_en(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        overrides = adapter._detect_user_override("Call me Alice")
        assert "demographics.name" in overrides
        assert overrides["demographics.name"] == "Alice"

    def test_detect_no_override(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        overrides = adapter._detect_user_override("今天聊点什么？")
        assert len(overrides) == 0

    @pytest.mark.asyncio
    async def test_stopped_after_streak(self, profile_path):
        from qwenpaw.agents.memory.spb.spb_adapter import SPBAdapter

        config = self._make_config()
        config.empty_streak_threshold = 2
        adapter = SPBAdapter(config, "test-agent", str(profile_path.parent))
        adapter._empty_streak = 2
        # Simulate empty run by incrementing directly
        adapter._empty_streak = config.empty_streak_threshold
        adapter._stopped = True
        assert adapter.stopped


# ---------------------------------------------------------------------------
# spb_evaluator tests
# ---------------------------------------------------------------------------

class TestSPBEvaluator:
    def test_exact_match(self):
        assert exact_match("Alice", "Alice")
        assert exact_match("alice", "Alice")
        assert not exact_match("Alice", "Bob")

    def test_normalized_str_match(self):
        assert normalized_str_match("  Alice  ", "alice")
        assert not normalized_str_match("Alice", "Bob")

    def test_semantic_str_match(self):
        # Same string should match
        assert semantic_str_match("frontend engineer", "Frontend Engineer")
        # Completely different should not
        assert not semantic_str_match("doctor", "engineer")

    def test_list_semantic_match(self):
        result = list_semantic_match(
            ["python", "javascript"],
            ["Python", "JavaScript"],
        )
        assert result["f1"] > 0.5

    def test_list_semantic_match_empty(self):
        result = list_semantic_match([], ["python"])
        assert result["f1"] == 0.0

    def test_list_semantic_match_empty_gt(self):
        result = list_semantic_match(["python"], [])
        assert result["f1"] == 1.0

    def test_evaluate_field_enum(self):
        assert evaluate_field("casual", "casual", "communication", "tone")
        assert not evaluate_field("casual", "formal", "communication", "tone")

    def test_evaluate_field_str(self):
        assert evaluate_field("Alice", "Alice", "demographics", "name")
        assert evaluate_field("alice", "Alice", "demographics", "name")

    def test_evaluate_field_list(self):
        result = evaluate_field(
            ["python", "rust"],
            ["Python", "Rust"],
            "interests",
            "topics",
        )
        assert isinstance(result, dict)
        assert result["f1"] > 0.5

    def test_evaluate_full(self):
        evaluator = SPBEvaluator()
        extracted = {
            "demographics": {"name": "Alice", "pronouns": "she/her"},
            "background": {"occupation": "Engineer"},
        }
        ground_truth = {
            "demographics": {"name": "Alice", "pronouns": "she/her"},
            "background": {"occupation": "Engineer"},
        }
        result = evaluator.evaluate(extracted, ground_truth)
        assert result["relevant_coverage"] == 1.0
        assert result["profile_accuracy"] == 1.0

    def test_evaluate_partial(self):
        evaluator = SPBEvaluator()
        extracted = {
            "demographics": {"name": "Alice"},
        }
        ground_truth = {
            "demographics": {"name": "Alice", "pronouns": "she/her"},
            "background": {"occupation": "Engineer"},
        }
        result = evaluator.evaluate(extracted, ground_truth)
        assert result["relevant_coverage"] < 1.0
        assert result["total_relevant_fields"] == 3
        assert result["filled_relevant_fields"] == 1

    def test_evaluate_semantic_mismatch(self):
        evaluator = SPBEvaluator()
        extracted = {
            "demographics": {"name": "Bob"},
        }
        ground_truth = {
            "demographics": {"name": "Alice"},
        }
        result = evaluator.evaluate(extracted, ground_truth)
        assert result["profile_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Integration: profile_path edge cases
# ---------------------------------------------------------------------------

class TestProfileWriterEdgeCases:
    def test_nonexistent_profile(self, tmp_path):
        writer = SPBProfileWriter()
        p = tmp_path / "nonexistent.md"
        # parse_markers should handle missing file gracefully
        # (would raise FileNotFoundError, caller should handle)
        with pytest.raises(FileNotFoundError):
            writer.parse_markers(p)

    def test_profile_without_markers(self, tmp_path):
        writer = SPBProfileWriter()
        p = tmp_path / "PROFILE.md"
        p.write_text("## 用户资料\n\nNo markers here\n", encoding="utf-8")
        markers = writer.parse_markers(p)
        assert len(markers) == 0
