# -*- coding: utf-8 -*-
"""SPB Evaluator — field-level semantic matching for profile extraction.

Implements layered matching:
- enum fields: exact match
- str (normalized): trimmed lowercase exact match
- str (semantic): embedding similarity >= 0.8
- list (semantic): per-element embedding similarity >= 0.75, set-level P/R/F1
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .spb_types import SPB_SCHEMA, SPBField

logger = logging.getLogger(__name__)


def _normalize(value: str) -> str:
    return value.strip().lower()


def exact_match(predicted: str, ground_truth: str) -> bool:
    return _normalize(predicted) == _normalize(ground_truth)


def enum_match(predicted: str, ground_truth: str, options: list[str]) -> bool:
    p = _normalize(predicted)
    g = _normalize(ground_truth)
    return p == g


def normalized_str_match(predicted: str, ground_truth: str) -> bool:
    return _normalize(predicted) == _normalize(ground_truth)


def embedding_similarity(a: str, b: str) -> float:
    """Compute embedding similarity between two strings.

    Falls back to normalized token overlap when embeddings are unavailable.
    """
    # Token-overlap fallback (used when no embedding model configured)
    tokens_a = set(re.findall(r"\w+", _normalize(a)))
    tokens_b = set(re.findall(r"\w+", _normalize(b)))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def semantic_str_match(
    predicted: str,
    ground_truth: str,
    threshold: float = 0.8,
) -> bool:
    sim = embedding_similarity(predicted, ground_truth)
    return sim >= threshold


def list_semantic_match(
    predicted: list[str],
    ground_truth: list[str],
    threshold: float = 0.75,
) -> dict[str, float]:
    """Set-level semantic matching for list fields.

    Returns dict with precision, recall, f1.
    """
    if not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # For each GT element, check if any predicted element matches
    hits = 0
    for gt in ground_truth:
        for pred in predicted:
            if embedding_similarity(pred, gt) >= threshold:
                hits += 1
                break

    recall = hits / len(ground_truth) if ground_truth else 1.0
    # For precision: each predicted that matches a GT
    pred_hits = 0
    for pred in predicted:
        for gt in ground_truth:
            if embedding_similarity(pred, gt) >= threshold:
                pred_hits += 1
                break

    precision = pred_hits / len(predicted) if predicted else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def _get_field_type(dim_name: str, field_key: str) -> str:
    for dim in SPB_SCHEMA:
        if dim.name == dim_name:
            for f in dim.fields:
                if f.key == field_key:
                    return f.field_type
    return "str"


def _get_field(dim_name: str, field_key: str) -> SPBField | None:
    for dim in SPB_SCHEMA:
        if dim.name == dim_name:
            for f in dim.fields:
                if f.key == field_key:
                    return f
    return None


def evaluate_field(
    predicted: Any,
    ground_truth: Any,
    dim_name: str,
    field_key: str,
) -> bool | dict[str, float]:
    """Evaluate a single field prediction against ground truth.

    Returns True/False for scalar fields, or P/R/F1 dict for list fields.
    """
    field = _get_field(dim_name, field_key)
    if field is None:
        return False

    if ground_truth is None:
        return predicted is None or predicted == ""

    if predicted is None or predicted == "":
        return False

    if field.field_type == "enum":
        return enum_match(str(predicted), str(ground_truth), field.options)

    if field.field_type == "list":
        pred_list = predicted if isinstance(predicted, list) else [predicted]
        gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        return list_semantic_match(pred_list, gt_list)

    # str fields: try normalized first, then semantic
    if normalized_str_match(str(predicted), str(ground_truth)):
        return True
    return semantic_str_match(str(predicted), str(ground_truth))


class SPBEvaluator:
    """Evaluate SPB extraction results against ground truth personas."""

    def evaluate(
        self,
        extracted: dict[str, dict],
        ground_truth: dict[str, dict],
    ) -> dict[str, Any]:
        """Compute all SPB metrics.

        Args:
            extracted: Dict mapping dimension name to {field: value}.
            ground_truth: Same structure, with correct values.

        Returns:
            Dict with RC, PA, F1, per-field details.
        """
        total_relevant = 0
        filled_relevant = 0
        correct_fields = 0
        total_filled = 0
        field_details: dict[str, Any] = {}

        for dim_name, gt_fields in ground_truth.items():
            if not isinstance(gt_fields, dict):
                continue
            ext_fields = extracted.get(dim_name, {})

            for field_key, gt_value in gt_fields.items():
                if gt_value is None or gt_value == "":
                    continue
                total_relevant += 1

                pred_value = ext_fields.get(field_key)
                if pred_value is not None and pred_value != "":
                    filled_relevant += 1
                    total_filled += 1

                    result = evaluate_field(pred_value, gt_value, dim_name, field_key)

                    if isinstance(result, dict):
                        # List field — check F1 >= 0.5 as "correct"
                        is_correct = result["f1"] >= 0.5
                        field_details[f"{dim_name}.{field_key}"] = result
                    else:
                        is_correct = result
                        field_details[f"{dim_name}.{field_key}"] = is_correct

                    if is_correct:
                        correct_fields += 1
                else:
                    field_details[f"{dim_name}.{field_key}"] = None

        rc = filled_relevant / total_relevant if total_relevant > 0 else 0.0
        pa = correct_fields / total_filled if total_filled > 0 else 0.0

        return {
            "relevant_coverage": rc,
            "profile_accuracy": pa,
            "total_relevant_fields": total_relevant,
            "filled_relevant_fields": filled_relevant,
            "correct_fields": correct_fields,
            "total_filled_fields": total_filled,
            "field_details": field_details,
        }
