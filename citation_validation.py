"""Validate that model-produced citation labels exist in retrieved evidence."""
from __future__ import annotations

import re
from typing import Iterable

# Match case-insensitively so malformed labels such as [s1] are detected; IDs
# are compared case-sensitively against the available source IDs below.
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)


def validate_citations(answer: str, available_source_ids: Iterable[str]) -> dict:
    """Check source-label references, not whether cited text semantically proves a claim.

    Citation IDs are case-sensitive by design. Repeated IDs are returned once
    in order of appearance. A response with no citation labels is not valid.
    """
    found = list(dict.fromkeys(_CITATION_PATTERN.findall(str(answer or ""))))
    available = {str(source_id) for source_id in available_source_ids}
    invalid = [source_id for source_id in found if source_id not in available]
    valid_used = [source_id for source_id in found if source_id in available]
    return {
        "has_citations": bool(found),
        "valid": bool(found) and not invalid,
        "used_source_ids": valid_used,
        "invalid_source_ids": invalid,
        "available_source_ids": sorted(
            available,
            key=lambda item: (
                item[0] if item else "",
                int(item[1:]) if len(item) > 1 and item[1:].isdigit() else 0,
            ),
        ),
    }
