"""Small, dependency-free helpers for timestamp-aware YouTube RAG.

Keeping URL validation, timestamp formatting, and transcript chunking separate
makes the core behavior testable without network access or paid API keys.
"""
from __future__ import annotations

import math
import re
from urllib.parse import parse_qs, urlparse

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(value: str | None) -> str | None:
    """Return a validated YouTube video ID from common URL forms or a raw ID."""
    text = str(value or "").strip()
    if VIDEO_ID_PATTERN.fullmatch(text):
        return text

    candidate = text if "://" in text else "https://" + text
    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None

    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
                  "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            match = re.match(r"^/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})(?:/|$)", parsed.path)
            if match:
                video_id = match.group(1)

    return video_id if video_id and VIDEO_ID_PATTERN.fullmatch(video_id) else None


def format_timestamp(seconds: float | int | None) -> str:
    """Render seconds as MM:SS or HH:MM:SS without accepting NaN/negative values."""
    try:
        total = max(0, int(float(seconds or 0)))
        if not math.isfinite(float(seconds or 0)):
            total = 0
    except (TypeError, ValueError, OverflowError):
        total = 0

    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"



_TIMESTAMP_LINE = re.compile(
    r"^\s*\[?(?P<timestamp>(?:\d{1,2}:)?\d{1,2}:\d{2})\]?\s*(?P<text>.*)$"
)


def _parse_timestamp(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError("unsupported timestamp")


def parse_pasted_transcript(value: str) -> tuple[list[dict], bool]:
    """Parse pasted transcript lines, preserving timestamps when present.

    Supports inline lines such as 0:12 spoken words and alternating 0:12 /
    spoken words lines. When no timestamp lines exist, the complete text is
    treated as an un-timestamped transcript. Returns snippets and a time flag.
    """
    raw = str(value or "").strip()
    if not raw:
        return [], False

    segments: list[dict] = []
    pending_start: float | None = None
    timestamp_count = 0

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _TIMESTAMP_LINE.match(stripped)
        if match:
            try:
                start = _parse_timestamp(match.group("timestamp"))
            except (ValueError, TypeError):
                match = None
        if match:
            timestamp_count += 1
            text = match.group("text").strip()
            if text:
                segments.append({"text": text, "start": start, "duration": 0.0})
                pending_start = None
            else:
                pending_start = start
            continue

        if pending_start is not None:
            segments.append({"text": stripped, "start": pending_start, "duration": 0.0})
            pending_start = None
        elif segments:
            segments[-1]["text"] += " " + stripped

    has_timestamps = timestamp_count > 0
    if not segments:
        return [{"text": " ".join(raw.split()), "start": 0.0, "duration": 0.0}], False

    segments = sorted(segments, key=lambda segment: segment["start"])
    for index, segment in enumerate(segments[:-1]):
        next_start = segments[index + 1]["start"]
        segment["duration"] = max(0.0, next_start - segment["start"])

    if not has_timestamps:
        return [{"text": " ".join(raw.split()), "start": 0.0, "duration": 0.0}], False

    return segments, True

def _value(item, key: str, default=0):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def _split_oversized_segment(segment: dict, max_chars: int) -> list[dict]:
    text = segment["text"]
    if len(text) <= max_chars:
        return [segment]

    pieces = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append({**segment, "text": current})
            current = ""
        if len(word) > max_chars:
            if current:
                pieces.append({**segment, "text": current})
                current = ""
            for offset in range(0, len(word), max_chars):
                pieces.append({**segment, "text": word[offset:offset + max_chars]})
        else:
            current = word if not current else current + " " + word
    if current:
        pieces.append({**segment, "text": current})
    return pieces


def build_timestamped_chunks(
    snippets,
    max_chars: int = 1400,
    overlap_chars: int = 180,
) -> list[dict]:
    """Group timestamped transcript snippets into overlapping, timestamped chunks.

    Each result contains text, start_seconds, end_seconds, and chunk_index.
    Overlap retains whole caption snippets so timestamps remain meaningful.
    """
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars - 1")

    segments = []
    for snippet in snippets or []:
        text = " ".join(str(_value(snippet, "text", "") or "").split())
        if not text:
            continue
        start = _safe_float(_value(snippet, "start", 0))
        duration = _safe_float(_value(snippet, "duration", 0))
        base = {"text": text, "start": start, "duration": duration}
        segments.extend(_split_oversized_segment(base, max_chars))

    chunks = []
    buffer = []
    buffer_chars = 0

    def emit(items):
        if not items:
            return
        chunks.append({
            "text": " ".join(item["text"] for item in items).strip(),
            "start_seconds": round(items[0]["start"], 3),
            "end_seconds": round(max(item["start"] + item["duration"] for item in items), 3),
            "chunk_index": len(chunks),
        })

    for segment in segments:
        segment_chars = len(segment["text"])
        if buffer and buffer_chars + 1 + segment_chars > max_chars:
            emit(buffer)
            carry = []
            carry_chars = 0
            for previous in reversed(buffer):
                previous_chars = len(previous["text"]) + (1 if carry else 0)
                if carry and carry_chars + previous_chars > overlap_chars:
                    break
                if not carry and len(previous["text"]) > overlap_chars:
                    break
                carry.insert(0, previous)
                carry_chars += previous_chars
                if carry_chars >= overlap_chars:
                    break
            buffer = carry
            buffer_chars = sum(len(item["text"]) for item in buffer) + max(0, len(buffer) - 1)
            if buffer and buffer_chars + 1 + segment_chars > max_chars:
                buffer = []
                buffer_chars = 0

        if buffer:
            buffer_chars += 1
        buffer.append(segment)
        buffer_chars += segment_chars

    emit(buffer)
    return [chunk for chunk in chunks if chunk["text"]]
