from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json


TRANSCRIPT_CHUNK_CHAR_LIMIT = 12000


@dataclass(frozen=True)
class TranscriptSegment:
    source_id: str
    start: float
    end: float
    speaker: str
    text: str

    def render(self) -> str:
        return (
            f"[{self.source_id} {format_timestamp(self.start)}-"
            f"{format_timestamp(self.end)} | {self.speaker}] {self.text}"
        )


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def read_transcript(
    transcript_file: Path, character_map: dict[str, str]
) -> list[TranscriptSegment]:
    segments = []
    with transcript_file.open(encoding="utf-8") as transcript:
        for line in transcript:
            item = json.loads(line)
            text = " ".join(item.get("text", "").split())
            if not text:
                continue
            user_id = item.get("user_id")
            speaker = (
                character_map.get(str(user_id), str(user_id))
                if user_id is not None
                else "Unknown speaker"
            )
            segments.append(
                TranscriptSegment(
                    source_id=f"S{len(segments) + 1:04}",
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    speaker=speaker,
                    text=text,
                )
            )
    return segments


def chunk_transcript(
    segments: Sequence[TranscriptSegment],
    max_chars: int = TRANSCRIPT_CHUNK_CHAR_LIMIT,
    overlap_segments: int = 1,
) -> list[list[TranscriptSegment]]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    if overlap_segments < 0:
        raise ValueError("overlap_segments must not be negative.")

    chunks = []
    start = 0
    while start < len(segments):
        end = start
        chunk_chars = 0
        while end < len(segments):
            segment_chars = len(segments[end].render())
            separator_chars = 1 if end > start else 0
            if end > start and chunk_chars + separator_chars + segment_chars > max_chars:
                break
            chunk_chars += separator_chars + segment_chars
            end += 1

        chunks.append(list(segments[start:end]))
        if end == len(segments):
            break
        start = max(start + 1, end - overlap_segments)

    return chunks
