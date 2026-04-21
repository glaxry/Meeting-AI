from __future__ import annotations

import numpy as np

from meeting_ai.config import get_settings
from meeting_ai.streaming import FunASRStreamingTranscriber, decode_pcm16_base64, encode_pcm16_base64


class FakeStreamingModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        text = self.outputs[index] if index < len(self.outputs) else ""
        return [{"text": text, "tokens": []}]


def test_streaming_session_accumulates_partial_text_and_marks_final() -> None:
    settings = get_settings().model_copy(
        update={
            "use_gpu": False,
            "funasr_streaming_model": "paraformer-zh-streaming",
            "funasr_streaming_chunk_size": "0,10,5",
        }
    )
    fake_model = FakeStreamingModel(["欢迎", "大家"])
    transcriber = FunASRStreamingTranscriber(settings=settings, model_factory=lambda: fake_model)
    session = transcriber.create_session(language="zh", sample_rate=16000, session_id="stream-demo")

    first_event = session.process_chunk(np.ones(3200, dtype=np.float32) * 0.2, sample_rate=16000, is_final=False)
    final_event = session.process_chunk(np.ones(3200, dtype=np.float32) * 0.2, sample_rate=16000, is_final=True)

    assert first_event.delta_text == "欢迎"
    assert first_event.cumulative_text == "欢迎"
    assert final_event.delta_text == "大家"
    assert final_event.cumulative_text == "欢迎大家"
    assert final_event.is_final is True
    assert fake_model.calls[1]["is_final"] is True


def test_streaming_session_resamples_audio_before_inference() -> None:
    settings = get_settings().model_copy(update={"use_gpu": False, "streaming_target_sample_rate": 16000})
    fake_model = FakeStreamingModel(["hello"])
    transcriber = FunASRStreamingTranscriber(settings=settings, model_factory=lambda: fake_model)
    session = transcriber.create_session(language="en", sample_rate=8000)

    event = session.process_chunk(np.ones(8000, dtype=np.float32) * 0.1, sample_rate=8000, is_final=False)

    model_input = fake_model.calls[0]["input"]
    assert isinstance(model_input, np.ndarray)
    assert model_input.shape[0] == 16000
    assert event.received_seconds == 1.0
    assert event.metadata["resampled"] is True


def test_pcm16_base64_round_trip_preserves_waveform() -> None:
    audio = np.array([0.0, 0.25, -0.25, 0.75], dtype=np.float32)

    encoded = encode_pcm16_base64(audio)
    decoded = decode_pcm16_base64(encoded)

    np.testing.assert_allclose(decoded, audio, atol=2.0 / 32768.0)
