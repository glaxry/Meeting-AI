from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import websockets


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_ai.streaming import encode_pcm16_base64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an audio file to the Meeting AI streaming WebSocket endpoint.")
    parser.add_argument("--audio", required=True, help="Path to a wav/flac/etc. audio file.")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream/transcribe", help="Streaming WebSocket URL.")
    parser.add_argument("--language", default="zh", help="Language hint for the streaming session.")
    parser.add_argument("--chunk-seconds", type=float, default=2.0, help="Chunk duration in seconds.")
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Sleep chunk-seconds between sends to mimic live capture.",
    )
    parser.add_argument(
        "--print-cumulative",
        action="store_true",
        help="Print the full cumulative transcript after every chunk.",
    )
    parser.add_argument("--session-id", help="Optional custom session id.")
    return parser


async def run_demo(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    chunk_samples = max(1, int(args.chunk_seconds * sample_rate))
    async with websockets.connect(args.url, max_size=None) as websocket:
        start_payload = {
            "type": "start",
            "language": args.language,
            "sample_rate": sample_rate,
        }
        if args.session_id:
            start_payload["session_id"] = args.session_id
        await websocket.send(json.dumps(start_payload, ensure_ascii=False))
        ready_message = json.loads(await websocket.recv())
        print(json.dumps(ready_message, ensure_ascii=False, indent=2))

        for chunk_index, start in enumerate(range(0, audio.shape[0], chunk_samples)):
            chunk = audio[start : start + chunk_samples]
            is_final = start + chunk_samples >= audio.shape[0]
            payload = {
                "type": "chunk",
                "audio_base64": encode_pcm16_base64(chunk),
                "sample_rate": sample_rate,
                "is_final": is_final,
            }

            started = time.perf_counter()
            await websocket.send(json.dumps(payload))
            message = json.loads(await websocket.recv())
            elapsed = round(time.perf_counter() - started, 3)

            if message.get("event") == "error":
                raise RuntimeError(message.get("detail", "Unknown streaming error."))

            transcript = message.get("transcript", {})
            print(
                f"[chunk {chunk_index:02d}] event={message.get('event')} latency={elapsed:.3f}s "
                f"delta={transcript.get('delta_text', '')!r}"
            )
            if args.print_cumulative:
                print(transcript.get("cumulative_text", ""))

            if args.simulate_realtime and not is_final:
                await asyncio.sleep(args.chunk_seconds)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
