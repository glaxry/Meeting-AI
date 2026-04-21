from __future__ import annotations

import base64
import binascii
import logging
import math
import re
import threading
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
from funasr import AutoModel
from scipy.signal import resample_poly

from .config import MeetingAISettings, get_settings
from .schemas import StreamingSessionInfo, StreamingTranscriptEvent


LOGGER = logging.getLogger(__name__)
_PCM16_SCALE = 32768.0
_ASCII_TRAILING = re.compile(r"[A-Za-z0-9]$")
_ASCII_LEADING = re.compile(r"^[A-Za-z0-9]")


def _streaming_model_kwargs(settings: MeetingAISettings) -> dict[str, Any]:
    return {
        "model": settings.funasr_streaming_model,
        "device": settings.device,
        "hub": settings.funasr_hub,
        "disable_update": True,
    }


def _normalize_audio_chunk(audio_chunk: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_chunk)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    if audio.ndim == 2:
        if audio.shape[0] <= audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError("Audio chunk must be mono after normalization.")

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)

    return np.clip(audio, -1.0, 1.0)


def _resample_audio(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = source_sample_rate // divisor
    return resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)


def _merge_stream_text(existing_text: str, delta_text: str) -> str:
    if not delta_text:
        return existing_text
    if not existing_text:
        return delta_text
    if existing_text.endswith((" ", "\n")) or delta_text.startswith((" ", "\n", ".", ",", "!", "?", ";", ":")):
        return f"{existing_text}{delta_text}"
    if _ASCII_TRAILING.search(existing_text) and _ASCII_LEADING.search(delta_text):
        return f"{existing_text} {delta_text}"
    return f"{existing_text}{delta_text}"


def encode_pcm16_base64(audio_chunk: np.ndarray) -> str:
    normalized = _normalize_audio_chunk(audio_chunk)
    pcm = np.clip(normalized * (_PCM16_SCALE - 1), -_PCM16_SCALE, _PCM16_SCALE - 1).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def decode_pcm16_base64(audio_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(audio_base64)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 must be valid base64-encoded PCM16 data.") from exc
    if len(raw) % 2 != 0:
        raise ValueError("PCM16 payload length must be divisible by 2.")
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _PCM16_SCALE).copy()


class StreamingASRSession:
    def __init__(
        self,
        *,
        model: AutoModel,
        settings: MeetingAISettings,
        language: str,
        sample_rate: int,
        session_id: str | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.language = language
        self.sample_rate = int(sample_rate)
        self.target_sample_rate = int(settings.streaming_target_sample_rate)
        self.chunk_size = settings.parsed_funasr_streaming_chunk_size
        self.encoder_chunk_look_back = settings.funasr_streaming_encoder_chunk_look_back
        self.decoder_chunk_look_back = settings.funasr_streaming_decoder_chunk_look_back
        self.session_id = session_id or uuid.uuid4().hex
        self.cache: dict[str, Any] = {}
        self.chunk_index = 0
        self.total_target_samples = 0
        self.cumulative_text = ""
        self.closed = False

    def session_info(self) -> StreamingSessionInfo:
        return StreamingSessionInfo(
            session_id=self.session_id,
            language=self.language,
            sample_rate=self.sample_rate,
            target_sample_rate=self.target_sample_rate,
            asr_model=self.settings.funasr_streaming_model,
            chunk_size=list(self.chunk_size),
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
            metadata={
                "device": self.settings.device,
                "hub": self.settings.funasr_hub,
            },
        )

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        *,
        sample_rate: int | None = None,
        is_final: bool = False,
    ) -> StreamingTranscriptEvent:
        if self.closed:
            raise RuntimeError(f"Streaming session {self.session_id} is already closed.")

        effective_sample_rate = int(sample_rate or self.sample_rate)
        normalized = _normalize_audio_chunk(audio_chunk)
        resampled = _resample_audio(normalized, effective_sample_rate, self.target_sample_rate)
        self.total_target_samples += int(resampled.shape[0])

        payload: dict[str, Any] = {}
        delta_text = ""
        if resampled.size:
            result = self.model.generate(
                input=resampled,
                cache=self.cache,
                is_final=is_final,
                chunk_size=list(self.chunk_size),
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            if isinstance(result, list) and result:
                payload = result[0] or {}
            elif isinstance(result, dict):
                payload = result
            delta_text = str(payload.get("text", "") or "").strip()

        if delta_text:
            self.cumulative_text = _merge_stream_text(self.cumulative_text, delta_text)

        event = StreamingTranscriptEvent(
            session_id=self.session_id,
            chunk_index=self.chunk_index,
            delta_text=delta_text,
            cumulative_text=self.cumulative_text,
            is_final=is_final,
            received_seconds=round(self.total_target_samples / float(self.target_sample_rate), 3),
            sample_rate=effective_sample_rate,
            target_sample_rate=self.target_sample_rate,
            metadata={
                "chunk_samples": int(normalized.shape[0]),
                "resampled_chunk_samples": int(resampled.shape[0]),
                "raw_result_keys": sorted(payload.keys()),
                "chunk_size": list(self.chunk_size),
                "resampled": effective_sample_rate != self.target_sample_rate,
                "language": self.language,
            },
        )
        self.chunk_index += 1
        if is_final:
            self.closed = True
        return event

    def snapshot(self, *, is_final: bool = False) -> StreamingTranscriptEvent:
        if is_final:
            self.closed = True
        return StreamingTranscriptEvent(
            session_id=self.session_id,
            chunk_index=max(self.chunk_index - 1, 0),
            delta_text="",
            cumulative_text=self.cumulative_text,
            is_final=is_final,
            received_seconds=round(self.total_target_samples / float(self.target_sample_rate), 3),
            sample_rate=self.sample_rate,
            target_sample_rate=self.target_sample_rate,
            metadata={
                "chunk_size": list(self.chunk_size),
                "language": self.language,
                "snapshot_only": True,
            },
        )


class FunASRStreamingTranscriber:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        model_factory: Callable[[], AutoModel] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model_factory = model_factory
        self._model: AutoModel | None = None

    def _load(self) -> AutoModel:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                LOGGER.info(
                    "Loading FunASR streaming model=%s on %s",
                    self.settings.funasr_streaming_model,
                    self.settings.device,
                )
                self._model = AutoModel(**_streaming_model_kwargs(self.settings))
        return self._model

    def create_session(
        self,
        *,
        language: str = "zh",
        sample_rate: int | None = None,
        session_id: str | None = None,
    ) -> StreamingASRSession:
        return StreamingASRSession(
            model=self._load(),
            settings=self.settings,
            language=language,
            sample_rate=sample_rate or self.settings.streaming_target_sample_rate,
            session_id=session_id,
        )


class StreamingSessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, StreamingASRSession] = {}

    def add(self, session: StreamingASRSession) -> StreamingASRSession:
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> StreamingASRSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def pop(self, session_id: str) -> StreamingASRSession | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

