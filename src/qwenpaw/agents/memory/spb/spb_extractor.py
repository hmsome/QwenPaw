# -*- coding: utf-8 -*-
"""SPB Extractor — per-dimension LLM extraction.

Uses ``create_model_and_formatter(agent_id)`` to create LLM instances,
which auto-wraps with ``TokenRecordingModelWrapper`` for token tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agentscope.message import Msg

from .spb_types import SPBDimension
from .spb_prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


class SPBExtractor:
    """Extracts profile fields from conversation using per-dimension LLM calls."""

    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._model = None
        self._formatter = None

    async def _ensure_model(self):
        if self._model is None:
            from ..model_factory import create_model_and_formatter

            self._model, self._formatter = create_model_and_formatter(
                self._agent_id,
            )

    async def extract_dimension(
        self,
        messages_text: str,
        dimension: SPBDimension,
        language: str = "en",
    ) -> dict[str, Any]:
        """Run a single LLM extraction for one dimension.

        Args:
            messages_text: Recent conversation text.
            dimension: SPBDimension to extract.
            language: Language code.

        Returns:
            Dict mapping field keys to extracted values,
            or ``{"extracted": False}`` if nothing found.
        """
        await self._ensure_model()

        prompt = build_extraction_prompt(messages_text, dimension, language)
        messages = [Msg(role="user", content=prompt, name="user")]

        try:
            formatted = await self._formatter._format(messages)
            response = await self._model(formatted)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON from response
            result = self._parse_json_response(content)
            if result and result.get("extracted") is False:
                return {}
            return result or {}
        except Exception as e:
            logger.warning("SPB extraction failed for %s: %s", dimension.name, e)
            return {}

    async def extract_all(
        self,
        messages: list[Msg],
        unfilled_dimensions: list[SPBDimension],
        language: str = "en",
    ) -> dict[str, dict]:
        """Extract all unfilled dimensions in parallel.

        Args:
            messages: Conversation messages.
            unfilled_dimensions: Dimensions with unfilled fields.
            language: Language code.

        Returns:
            Dict mapping dimension name to extraction result dict.
        """
        messages_text = self._messages_to_text(messages)
        if not messages_text.strip():
            return {}

        tasks = [
            self.extract_dimension(messages_text, dim, language)
            for dim in unfilled_dimensions
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[str, dict] = {}
        for dim, result in zip(unfilled_dimensions, results_list):
            if isinstance(result, Exception):
                logger.warning(
                    "SPB extract_all error for %s: %s", dim.name, result,
                )
                continue
            if result:  # Non-empty dict means something was extracted
                results[dim.name] = result
        return results

    @staticmethod
    def _messages_to_text(messages: list[Msg]) -> str:
        """Convert message list to plain text for extraction."""
        parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if isinstance(content, list):
                # Handle list content blocks (TextBlock etc.)
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    elif hasattr(block, "text"):
                        texts.append(block.text)
                    elif isinstance(block, str):
                        texts.append(block)
                content = " ".join(texts)
            if content and isinstance(content, str):
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        """Extract JSON object from LLM response text."""
        # Try to find JSON in markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find raw JSON object
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
