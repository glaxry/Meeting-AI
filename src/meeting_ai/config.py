from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_secret_file(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


class MeetingAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    qwen_api_key: str | None = Field(default=None, alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_BASE_URL",
    )
    qwen_model: str = Field(default="qwen-plus", alias="QWEN_MODEL")

    huggingface_token: str | None = Field(default=None, alias="HUGGINGFACE_TOKEN")
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1", alias="PYANNOTE_MODEL")
    sentiment_transformer_model: str = Field(
        default="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        alias="SENTIMENT_TRANSFORMER_MODEL",
    )
    embedding_model: str = Field(default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL")

    funasr_model: str = Field(default="paraformer-zh", alias="FUNASR_MODEL")
    funasr_vad_model: str = Field(default="fsmn-vad", alias="FUNASR_VAD_MODEL")
    funasr_punc_model: str = Field(default="ct-punc-c", alias="FUNASR_PUNC_MODEL")
    funasr_hub: str = Field(default="ms", alias="FUNASR_HUB")

    use_gpu: bool = Field(default=True, alias="USE_GPU")

    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_retry_backoff_seconds: float = Field(default=1.5, alias="LLM_RETRY_BACKOFF_SECONDS")
    summary_map_reduce_threshold: int = Field(default=500, alias="SUMMARY_MAP_REDUCE_THRESHOLD")
    summary_chunk_target_words: int = Field(default=350, alias="SUMMARY_CHUNK_TARGET_WORDS")

    deepseek_key_file: Path = Field(default=PROJECT_ROOT / "api-key-deepseek")
    default_output_dir: Path = Field(default=PROJECT_ROOT / "data" / "outputs")
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_PERSIST_DIR")
    gradio_server_name: str = Field(default="127.0.0.1", alias="GRADIO_SERVER_NAME")
    gradio_server_port: int = Field(default=7860, alias="GRADIO_SERVER_PORT")

    @property
    def device(self) -> str:
        if not self.use_gpu:
            return "cpu"
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def resolved_deepseek_api_key(self) -> str | None:
        return self.deepseek_api_key or _read_secret_file(self.deepseek_key_file)

    @property
    def resolved_qwen_api_key(self) -> str | None:
        return self.qwen_api_key

    def ensure_output_dir(self) -> Path:
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        return self.default_output_dir

    def ensure_chroma_dir(self) -> Path:
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_persist_dir

    def redacted_summary(self) -> dict[str, object]:
        return {
            "device": self.device,
            "deepseek_key_present": bool(self.resolved_deepseek_api_key),
            "qwen_key_present": bool(self.resolved_qwen_api_key),
            "huggingface_token_present": bool(self.huggingface_token),
            "funasr_model": self.funasr_model,
            "pyannote_model": self.pyannote_model,
            "sentiment_transformer_model": self.sentiment_transformer_model,
            "embedding_model": self.embedding_model,
            "chroma_persist_dir": str(self.chroma_persist_dir),
        }


@lru_cache(maxsize=1)
def get_settings() -> MeetingAISettings:
    return MeetingAISettings()
