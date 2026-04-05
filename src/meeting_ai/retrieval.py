from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import MeetingAISettings, get_settings
from .schemas import RetrievalRecord, SummaryResult, TranscriptResult


def _summary_to_document(summary: SummaryResult, transcript: TranscriptResult | None = None) -> str:
    lines = ["Topics:"]
    lines.extend(f"- {item}" for item in summary.topics or ["None recorded"])
    lines.append("Decisions:")
    lines.extend(f"- {item}" for item in summary.decisions or ["None recorded"])
    lines.append("Follow-ups:")
    lines.extend(f"- {item}" for item in summary.follow_ups or ["None recorded"])

    if transcript is not None:
        excerpt = transcript.full_text[:1200].strip()
        if excerpt:
            lines.append("Transcript excerpt:")
            lines.append(excerpt)

    return "\n".join(lines)


@dataclass
class SentenceTransformerEmbedder:
    settings: MeetingAISettings
    _model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model, device=self.settings.device)
        return self._model

    def encode_texts(self, texts: list[str], task: str = "passage") -> list[list[float]]:
        model = self._load()
        normalized_texts = texts
        if "e5" in self.settings.embedding_model.lower():
            prefix = "query: " if task == "query" else "passage: "
            normalized_texts = [f"{prefix}{text}" for text in texts]
        encoded = model.encode(
            normalized_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]


class MeetingVectorStore:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        collection_name: str = "meeting_history",
        collection: Any | None = None,
        embedder: SentenceTransformerEmbedder | Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._collection = collection
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings)

    def _get_collection(self) -> Any:
        if self._collection is None:
            import chromadb

            persist_path = self.settings.ensure_chroma_dir()
            client = chromadb.PersistentClient(path=str(persist_path))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_summary(
        self,
        summary: SummaryResult,
        transcript: TranscriptResult | None = None,
        meeting_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        collection = self._get_collection()
        resolved_meeting_id = meeting_id or uuid4().hex
        document = _summary_to_document(summary, transcript)
        embedding = self.embedder.encode_texts([document], task="passage")[0]
        base_metadata: dict[str, Any] = {
            "meeting_id": resolved_meeting_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if transcript is not None:
            base_metadata["audio_path"] = transcript.audio_path
            base_metadata["language"] = transcript.language
        if metadata:
            base_metadata.update(metadata)

        collection.upsert(
            ids=[resolved_meeting_id],
            documents=[document],
            metadatas=[base_metadata],
            embeddings=[embedding],
        )
        return resolved_meeting_id

    def query(self, question: str, top_k: int = 3) -> list[RetrievalRecord]:
        collection = self._get_collection()
        embedding = self.embedder.encode_texts([question], task="query")[0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        records: list[RetrievalRecord] = []

        for document, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
            payload = metadata or {}
            records.append(
                RetrievalRecord(
                    meeting_id=str(payload.get("meeting_id", "")),
                    document=str(document),
                    score=None if distance is None else round(1.0 - float(distance), 6),
                    metadata=payload,
                )
            )
        return records
