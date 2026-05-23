# -*- coding: utf-8 -*-
"""Profile writer for Structured Profile Bootstrapping (SPB).

Reads and updates PROFILE.md using HTML comment markers
(`<!-- spb:dimension.field:type -->`) to locate and replace field values.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(
    r"<!--\s*spb:(?P<dim>\w+)\.(?P<field>\w+):(?P<type>\w+)\s*-->"
)
_PENDING_PATTERNS = [
    re.compile(r"[*（]待补充）[*]"),
    re.compile(r"[*][(]pending[)]"),
    re.compile(r"[*][(]ожидает[)]"),
]
_PROFILE_SECTION_HEADERS = {"用户资料", "user profile", "профиль пользователя"}


def _is_pending(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    for pat in _PENDING_PATTERNS:
        if pat.search(v):
            return True
    return False


def _find_profile_section_range(lines: list[str]) -> tuple[int, int]:
    """Return (start, end) index range for the user profile section."""
    header_idx = None
    for i, line in enumerate(lines):
        low = line.strip().lower().lstrip("#").strip()
        if low in _PROFILE_SECTION_HEADERS:
            header_idx = i
            break
    if header_idx is None:
        return 0, len(lines)
    # Section extends to end of file or next ## heading
    end = len(lines)
    for j in range(header_idx + 1, len(lines)):
        if lines[j].startswith("## ") and j != header_idx:
            end = j
            break
    return header_idx, end


class SPBProfileWriter:
    """Writes structured profile fields into PROFILE.md via markers."""

    def parse_markers(self, profile_path: Path) -> dict[str, dict]:
        """Parse PROFILE.md and return marker info.

        Returns:
            Dict mapping ``"dimension.field"`` to ``{"line": int,
            "type": str, "value": str}``.
        """
        text = profile_path.read_text(encoding="utf-8")
        lines = text.split("\n")
        _, section_end = _find_profile_section_range(lines)

        results: dict[str, dict] = {}
        for i, line in enumerate(lines):
            if i > section_end:
                break
            m = _MARKER_RE.search(line)
            if not m:
                continue
            key = f"{m.group('dim')}.{m.group('field')}"
            # Value is the next non-empty line
            value = ""
            if i + 1 < len(lines):
                value = lines[i + 1].strip()
                # Strip leading markdown bold/label prefix: "- **Label:** value"
                # Extract the actual value part after the label
                label_match = re.match(r"[-*]\s*\*\*[^*]*\*\*\s*:?\s*(.*)", value)
                if label_match:
                    value = label_match.group(1).strip()
            results[key] = {
                "line": i,
                "type": m.group("type"),
                "value": value,
                "filled": not _is_pending(value),
            }
        return results

    def update_field(
        self,
        profile_path: Path,
        dimension: str,
        field: str,
        value: str,
    ) -> bool:
        """Update a single field in PROFILE.md.

        Returns True if the file was modified.
        """
        text = profile_path.read_text(encoding="utf-8")
        lines = text.split("\n")

        marker_pattern = f"<!-- spb:{dimension}.{field}:"
        for i, line in enumerate(lines):
            if marker_pattern in line:
                if i + 1 >= len(lines):
                    return False
                # Extract label prefix from next line
                next_line = lines[i + 1]
                label_match = re.match(r"([-*]\s*\*\*[^*]*\*\*\s*:?\s*)(.*)", next_line)
                if label_match:
                    prefix = label_match.group(1)
                    lines[i + 1] = f"{prefix}{value}"
                else:
                    lines[i + 1] = f"- {value}"

                profile_path.write_text("\n".join(lines), encoding="utf-8")
                logger.info("SPB updated %s.%s = %s", dimension, field, value)
                return True

        logger.debug("SPB marker not found for %s.%s", dimension, field)
        return False

    def apply_extraction_results(
        self,
        profile_path: Path,
        results: dict[str, dict],
    ) -> int:
        """Apply batch extraction results to PROFILE.md.

        Args:
            profile_path: Path to PROFILE.md.
            results: Dict mapping dimension name to extraction result dict,
                where each extraction result maps field keys to extracted
                values.

        Returns:
            Number of fields updated.
        """
        updated = 0
        markers = self.parse_markers(profile_path)
        for dim_name, fields in results.items():
            if not isinstance(fields, dict):
                continue
            for field_key, value in fields.items():
                if value is None or value == "" or value is False:
                    continue
                full_key = f"{dim_name}.{field_key}"
                marker_info = markers.get(full_key)
                if marker_info and not marker_info["filled"]:
                    if self.update_field(profile_path, dim_name, field_key, str(value)):
                        updated += 1
        return updated

    def get_unfilled_dimensions(
        self,
        profile_path: Path,
        schema: list,
    ) -> list:
        """Return dimensions that have at least one unfilled field.

        Args:
            profile_path: Path to PROFILE.md.
            schema: List of SPBDimension objects.

        Returns:
            List of SPBDimension objects with at least one unfilled field.
        """
        markers = self.parse_markers(profile_path)
        unfilled = []
        for dim in schema:
            for f in dim.fields:
                key = f"{dim.name}.{f.key}"
                info = markers.get(key)
                if info and not info["filled"]:
                    unfilled.append(dim)
                    break
        return unfilled
