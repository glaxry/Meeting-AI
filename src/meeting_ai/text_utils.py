from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import TranscriptResult, TranscriptSegment


JSON_START_CHARS = "{["
JSON_END_CHARS = {"{": "}", "[": "]"}


def transcript_to_segments(transcript: TranscriptResult | list[TranscriptSegment]) -> list[TranscriptSegment]:
    if isinstance(transcript, TranscriptResult):
        return transcript.segments
    return transcript


def transcript_to_text(transcript: TranscriptResult | list[TranscriptSegment]) -> str:
    segments = transcript_to_segments(transcript)
    return "\n".join(f"[{segment.speaker}] {segment.text}" for segment in segments)


def estimate_word_count(text: str) -> int:
    latin_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_tokens) + len(cjk_chars)


def chunk_transcript_segments(
    segments: list[TranscriptSegment],
    target_words: int,
) -> list[list[TranscriptSegment]]:
    if not segments:
        return []

    chunks: list[list[TranscriptSegment]] = []
    current_chunk: list[TranscriptSegment] = []
    current_words = 0

    for segment in segments:
        segment_words = max(estimate_word_count(segment.text), 1)
        if current_chunk and current_words + segment_words > target_words:
            chunks.append(current_chunk)
            current_chunk = []
            current_words = 0

        current_chunk.append(segment)
        current_words += segment_words

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_text(text: str, target_words: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        normalized = text.strip()
        return [normalized] if normalized else []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_words = 0

    for line in lines:
        line_words = max(estimate_word_count(line), 1)
        if current_lines and current_words + line_words > target_words:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_words = 0

        current_lines.append(line)
        current_words += line_words

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def parse_labelled_lines(text: str) -> list[TranscriptSegment]:
    pattern = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<body>.*)$")
    segments: list[TranscriptSegment] = []

    for index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            speaker = match.group("speaker").strip()
            body = match.group("body").strip()
        else:
            speaker = "SPEAKER_00"
            body = line
        if not body:
            continue
        segments.append(
            TranscriptSegment(
                speaker=speaker,
                text=body,
                start=float(index),
                end=float(index + 1),
            )
        )

    return segments


def load_transcript_json(path: str | Path) -> TranscriptResult:
    file_path = Path(path).expanduser().resolve()
    return TranscriptResult.model_validate_json(file_path.read_text(encoding="utf-8"))


def load_text_input(text: str | None = None, text_file: str | None = None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return Path(text_file).expanduser().resolve().read_text(encoding="utf-8")
    raise ValueError("Either text or text_file must be provided.")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_payload(text: str) -> Any:
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(stripped):
        if char not in JSON_START_CHARS:
            continue
        closing = JSON_END_CHARS[char]
        depth = 0
        for end_index in range(index, len(stripped)):
            if stripped[end_index] == char:
                depth += 1
            elif stripped[end_index] == closing:
                depth -= 1
                if depth == 0:
                    candidate = stripped[index : end_index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON payload found in model output.")


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated
