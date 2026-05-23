# -*- coding: utf-8 -*-
"""SPB Adapter — main integration point with 4-layer trigger strategy.

Implements adaptive + per-dimension early stopping trigger strategy:

  Layer 1: cheap pre-check (keyword → LLM fallback)
  Layer 2: per-dimension extraction (only unfilled dims)
  Layer 3: K-empty-streak global early stop
  Layer 4: user override detection (highest priority)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agentscope.message import Msg

from .spb_types import SPB_SCHEMA
from .spb_extractor import SPBExtractor
from .spb_profile_writer import SPBProfileWriter
from .spb_prompts import build_precheck_prompt

logger = logging.getLogger(__name__)

# Override detection patterns
_OVERRIDE_PATTERNS = [
    # Chinese: "我叫X" / "我的名字是X" / "叫我X"
    (re.compile(r"(?:我叫|我的名字(?:是|叫)|叫我)\s*['\"「」]?(\S+?)['\"「」]?(?:\s|[，。,.!！?？]|$)"), "demographics.name"),
    (re.compile(r"(?:我是)\s*(?:一名|一个|一位)?\s*(\S+?)(?:工程师|设计师|开发者|师|员|者)(?=\s|[，。,.!！?？]|$)"), "background.occupation"),
    # English: "call me X" / "my name is X" / "I'm X"
    (re.compile(r"(?:call me|my name is|i'm|i am)\s+(\w+)", re.IGNORECASE), "demographics.name"),
]


class SPBAdapter:
    """SPB main adapter implementing the 4-layer trigger strategy."""

    def __init__(self, config: Any, agent_id: str, working_dir: str):
        self._config = config
        self._agent_id = agent_id
        self._working_dir = working_dir
        self._profile_path = Path(working_dir) / "PROFILE.md"
        self._writer = SPBProfileWriter()
        self._extractor = SPBExtractor(agent_id)
        self._empty_streak = 0
        self._stopped = False

        # Filter dimensions if config specifies a subset
        if config.dimensions:
            self._schema = [d for d in SPB_SCHEMA if d.name in config.dimensions]
        else:
            self._schema = list(SPB_SCHEMA)

    @property
    def stopped(self) -> bool:
        return self._stopped

    async def should_extract(self, latest_user_msg: str) -> bool:
        """Layer 1: cheap pre-check.

        Returns True if the message might contain profile information.
        """
        if self._stopped:
            return False

        # Keyword check
        msg_lower = latest_user_msg.lower()
        for kw in self._config.pre_check_keywords_zh:
            if kw in msg_lower:
                return True
        for kw in self._config.pre_check_keywords_en:
            if kw.lower() in msg_lower:
                return True

        # LLM fallback (Layer 1 fallback)
        if self._config.pre_check_use_llm_fallback:
            return await self._llm_precheck(latest_user_msg)

        return False

    async def _llm_precheck(self, message: str) -> bool:
        """Lightweight LLM pre-check: does this message contain profile info?"""
        try:
            await self._extractor._ensure_model()
            prompt = build_precheck_prompt(message, language="en")
            messages = [Msg(role="user", content=prompt, name="user")]
            formatted = await self._extractor._formatter._format(messages)
            response = await self._extractor._model(formatted)
            content = response.content if hasattr(response, "content") else str(response)
            return "yes" in content.strip().lower()
        except Exception as e:
            logger.debug("SPB LLM precheck failed: %s", e)
            return False

    def _detect_user_override(self, latest_user_msg: str) -> dict[str, str]:
        """Layer 4: detect explicit user self-declarations.

        Returns a dict mapping "dimension.field" to the override value.
        """
        overrides: dict[str, str] = {}
        for pattern, key in _OVERRIDE_PATTERNS:
            m = pattern.search(latest_user_msg)
            if m:
                overrides[key] = m.group(1)
        return overrides

    async def run(self, all_messages: list[Msg]) -> dict[str, Any] | None:
        """Full extraction pipeline.

        Returns stats dict or None if nothing was done.
        """
        if self._stopped:
            return None

        if not self._profile_path.exists():
            logger.debug("SPB: PROFILE.md not found at %s", self._profile_path)
            return None

        latest_user_msg = self._get_latest_user_text(all_messages)
        if not latest_user_msg:
            return None

        # Layer 4: user override (highest priority)
        override_count = 0
        if self._config.enable_user_override:
            overrides = self._detect_user_override(latest_user_msg)
            for key, value in overrides.items():
                dim, field = key.split(".", 1)
                if self._writer.update_field(self._profile_path, dim, field, value):
                    override_count += 1
                    logger.info("SPB override: %s = %s", key, value)

        # Layer 1: pre-check
        if not await self.should_extract(latest_user_msg):
            if override_count > 0:
                return {
                    "updated_fields": override_count,
                    "token_cost": 0,
                    "stopped": False,
                }
            return None

        # Parse unfilled dimensions
        unfilled = self._writer.get_unfilled_dimensions(
            self._profile_path, self._schema,
        )
        if not unfilled:
            logger.info("SPB: all dimensions filled, stopping")
            self._stopped = True
            return {"updated_fields": 0, "token_cost": 0, "stopped": True}

        # Layer 2: per-dimension extraction
        language = self._detect_language(all_messages)
        results = await self._extractor.extract_all(
            all_messages, unfilled, language,
        )

        # Apply results
        updated = self._writer.apply_extraction_results(
            self._profile_path, results,
        )

        # Layer 3: K-empty-streak global early stop
        if updated == 0 and override_count == 0:
            self._empty_streak += 1
            if self._empty_streak >= self._config.empty_streak_threshold:
                self._stopped = True
                logger.info(
                    "SPB: %d consecutive empty extractions, stopping",
                    self._empty_streak,
                )
        else:
            self._empty_streak = 0

        return {
            "updated_fields": updated + override_count,
            "token_cost": 0,  # Tracked by TokenRecordingModelWrapper
            "stopped": self._stopped,
        }

    @staticmethod
    def _get_latest_user_text(messages: list[Msg]) -> str:
        """Get the most recent user message text."""
        for msg in reversed(messages):
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif hasattr(block, "text"):
                            texts.append(block.text)
                    return " ".join(texts)
                return str(content) if content else ""
        return ""

    @staticmethod
    def _detect_language(messages: list[Msg]) -> str:
        """Best-effort language detection from recent messages."""
        import re as _re

        for msg in reversed(messages):
            if msg.role != "user":
                continue
            text = msg.content if isinstance(msg.content, str) else ""
            if not text:
                continue
            # Count CJK characters
            cjk = len(_re.findall(r"[一-鿿]", text))
            if cjk > len(text) * 0.2:
                return "zh"
            return "en"
        return "en"
