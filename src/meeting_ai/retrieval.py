from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import MeetingAISettings, get_settings
from .schemas import RetrievalRecord, SummaryResult, TranscriptResult


_LEXICAL_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


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


def _transcript_window_to_document(transcript_window: list[Any]) -> str:
    return "\n".join(f"[{segment.speaker}] {segment.text}" for segment in transcript_window)


def _tokenize_lexical_text(text: str) -> list[str]:
    tokens = [token.lower() for token in _LEXICAL_TOKEN_PATTERN.findall(text or "")]
    if tokens:
        return tokens
    fallback = text.strip().lower()
    return [fallback] if fallback else []


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


@dataclass
class BM25LexicalRetriever:
    settings: MeetingAISettings

    def score_documents(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        from rank_bm25 import BM25Okapi

        tokenized_documents = [_tokenize_lexical_text(document) for document in documents]
        tokenized_query = _tokenize_lexical_text(query)
        if not tokenized_query:
            return [0.0 for _ in documents]
        index = BM25Okapi(tokenized_documents)
        return [float(score) for score in index.get_scores(tokenized_query)]


@dataclass
class CrossEncoderReranker:
    settings: MeetingAISettings
    _model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.retrieval_reranker_model, device=self.settings.device)
        return self._model

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        pairs = [(query, document) for document in documents]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


class MeetingVectorStore:
    def __init__(
        self,
        settings: MeetingAISettings | None = None,
        collection_name: str = "meeting_history",
        collection: Any | None = None,
        embedder: SentenceTransformerEmbedder | Any | None = None,
        lexical_retriever: BM25LexicalRetriever | Any | None = None,
        reranker: CrossEncoderReranker | Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._collection = collection
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings)
        self.lexical_retriever = lexical_retriever or BM25LexicalRetriever(self.settings)
        self.reranker = reranker or CrossEncoderReranker(self.settings)

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

    @staticmethod
    def _base_metadata(
        resolved_meeting_id: str,
        transcript: TranscriptResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_metadata: dict[str, Any] = {
            "meeting_id": resolved_meeting_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if transcript is not None:
            base_metadata["audio_path"] = transcript.audio_path
            base_metadata["language"] = transcript.language
        if metadata:
            base_metadata.update(metadata)
        return base_metadata

    @staticmethod
    def _build_where(chunk_type: str | None = None, meeting_id: str | None = None) -> dict[str, Any] | None:
        filters: list[dict[str, Any]] = []
        if chunk_type:
            filters.append({"chunk_type": {"$eq": chunk_type}})
        if meeting_id:
            filters.append({"meeting_id": {"$eq": meeting_id}})

        if len(filters) == 1:
            return filters[0]
        if len(filters) > 1:
            return {"$and": filters}
        return None

    @staticmethod
    def _iter_dense_results(result: dict[str, Any]) -> list[dict[str, Any]]:
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        if ids and ids[0] and not isinstance(ids[0], list):
            ids = [ids]
            documents = [documents]
            metadatas = [metadatas]
            distances = [distances]

        rows: list[dict[str, Any]] = []
        for item_id, document, metadata, distance in zip(ids[0], documents[0], metadatas[0], distances[0]):
            dense_score = None if distance is None else round(1.0 - float(distance), 6)
            rows.append(
                {
                    "id": str(item_id),
                    "document": str(document),
                    "metadata": dict(metadata or {}),
                    "dense_score": dense_score,
                }
            )
        return rows

    @staticmethod
    def _iter_collection_get(result: dict[str, Any]) -> list[dict[str, Any]]:
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        rows: list[dict[str, Any]] = []
        for item_id, document, metadata in zip(ids, documents, metadatas):
            rows.append(
                {
                    "id": str(item_id),
                    "document": str(document),
                    "metadata": dict(metadata or {}),
                }
            )
        return rows

    def _dense_search(
        self,
        question: str,
        *,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        collection = self._get_collection()
        embedding = self.embedder.encode_texts([question], task="query")[0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        return self._iter_dense_results(result)

    def _lexical_search(
        self,
        question: str,
        *,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        collection = self._get_collection()
        result = collection.get(where=where, include=["documents", "metadatas"])
        rows = self._iter_collection_get(result)
        documents = [row["document"] for row in rows]
        scores = self.lexical_retriever.score_documents(question, documents)
        ranked: list[dict[str, Any]] = []
        for row, score in zip(rows, scores):
            ranked.append({**row, "lexical_score": round(float(score), 6)})
        ranked.sort(key=lambda item: float(item.get("lexical_score") or 0.0), reverse=True)
        return ranked[: max(1, top_k)]

    def _merge_hybrid_candidates(
        self,
        dense_candidates: list[dict[str, Any]],
        lexical_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        rrf_k = max(1, int(self.settings.retrieval_rrf_k))

        def add_candidates(candidates: list[dict[str, Any]], source_name: str) -> None:
            for rank, candidate in enumerate(candidates, start=1):
                item_id = str(candidate["id"])
                merged_candidate = merged.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "document": candidate["document"],
                        "metadata": dict(candidate["metadata"]),
                        "dense_score": None,
                        "lexical_score": None,
                        "hybrid_seed_score": 0.0,
                        "sources": [],
                    },
                )
                merged_candidate["document"] = candidate["document"]
                merged_candidate["metadata"] = dict(candidate["metadata"])
                merged_candidate[f"{source_name}_score"] = candidate.get(f"{source_name}_score")
                merged_candidate["hybrid_seed_score"] = round(
                    float(merged_candidate["hybrid_seed_score"]) + (1.0 / (rrf_k + rank)),
                    6,
                )
                sources = set(merged_candidate["sources"])
                sources.add(source_name)
                merged_candidate["sources"] = sorted(sources)

        add_candidates(dense_candidates, "dense")
        add_candidates(lexical_candidates, "lexical")
        return sorted(
            merged.values(),
            key=lambda item: (
                float(item.get("hybrid_seed_score") or 0.0),
                float(item.get("dense_score") or 0.0),
                float(item.get("lexical_score") or 0.0),
            ),
            reverse=True,
        )

    def _apply_reranker(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not candidates or not self.settings.retrieval_enable_reranker:
            return candidates[:top_k], None

        rerank_pool_size = max(top_k, int(self.settings.retrieval_reranker_candidate_k))
        rerank_pool = candidates[:rerank_pool_size]
        try:
            scores = self.reranker.score_pairs(question, [candidate["document"] for candidate in rerank_pool])
        except Exception as exc:  # pragma: no cover
            return candidates[:top_k], str(exc)

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(rerank_pool, scores):
            reranked.append({**candidate, "reranker_score": round(float(score), 6)})
        reranked.sort(key=lambda item: float(item.get("reranker_score") or 0.0), reverse=True)
        return reranked[:top_k], None

    def add_meeting_chunks(
        self,
        transcript: TranscriptResult,
        summary: SummaryResult,
        meeting_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        collection = self._get_collection()
        resolved_meeting_id = meeting_id or uuid4().hex
        base_metadata = self._base_metadata(
            resolved_meeting_id=resolved_meeting_id,
            transcript=transcript,
            metadata=metadata,
        )

        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        documents.append(_summary_to_document(summary))
        ids.append(f"{resolved_meeting_id}_summary")
        metadatas.append(
            {
                **base_metadata,
                "chunk_type": "summary",
            }
        )

        chunk_size = max(1, int(self.settings.retrieval_chunk_size))
        for chunk_index, start_index in enumerate(range(0, len(transcript.segments), chunk_size)):
            transcript_window = transcript.segments[start_index : start_index + chunk_size]
            if not transcript_window:
                continue
            documents.append(_transcript_window_to_document(transcript_window))
            ids.append(f"{resolved_meeting_id}_chunk_{chunk_index}")
            metadatas.append(
                {
                    **base_metadata,
                    "chunk_type": "transcript",
                    "chunk_index": chunk_index,
                    "start_seconds": transcript_window[0].start,
                    "end_seconds": transcript_window[-1].end,
                    "speakers": sorted({segment.speaker for segment in transcript_window}),
                }
            )

        embeddings = self.embedder.encode_texts(documents, task="passage")
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return resolved_meeting_id

    def add_summary(
        self,
        summary: SummaryResult,
        transcript: TranscriptResult | None = None,
        meeting_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if transcript is not None and transcript.segments:
            return self.add_meeting_chunks(
                transcript=transcript,
                summary=summary,
                meeting_id=meeting_id,
                metadata=metadata,
            )

        collection = self._get_collection()
        resolved_meeting_id = meeting_id or uuid4().hex
        document = _summary_to_document(summary)
        embedding = self.embedder.encode_texts([document], task="passage")[0]
        base_metadata = self._base_metadata(
            resolved_meeting_id=resolved_meeting_id,
            transcript=transcript,
            metadata=metadata,
        )

        collection.upsert(
            ids=[f"{resolved_meeting_id}_summary"],
            documents=[document],
            metadatas=[{**base_metadata, "chunk_type": "summary"}],
            embeddings=[embedding],
        )
        return resolved_meeting_id

    def query(
        self,
        question: str,
        top_k: int = 3,
        chunk_type: str | None = None,
        meeting_id: str | None = None,
        strategy: str | None = None,
        use_reranker: bool | None = None,
    ) -> list[RetrievalRecord]:
        requested_top_k = max(1, int(top_k))
        requested_strategy = (strategy or self.settings.retrieval_strategy).strip().lower()
        reranker_enabled = self.settings.retrieval_enable_reranker if use_reranker is None else bool(use_reranker)
        where = self._build_where(chunk_type=chunk_type, meeting_id=meeting_id)

        reranker_error: str | None = None
        if requested_strategy == "dense":
            candidates = self._dense_search(
                question,
                top_k=max(requested_top_k, int(self.settings.retrieval_dense_candidate_k)),
                where=where,
            )
        elif requested_strategy == "lexical":
            candidates = self._lexical_search(
                question,
                top_k=max(requested_top_k, int(self.settings.retrieval_sparse_candidate_k)),
                where=where,
            )
        elif requested_strategy == "hybrid":
            dense_candidates = self._dense_search(
                question,
                top_k=max(requested_top_k, int(self.settings.retrieval_dense_candidate_k)),
                where=where,
            )
            lexical_candidates = self._lexical_search(
                question,
                top_k=max(requested_top_k, int(self.settings.retrieval_sparse_candidate_k)),
                where=where,
            )
            candidates = self._merge_hybrid_candidates(dense_candidates, lexical_candidates)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {requested_strategy}")

        if reranker_enabled:
            candidates, reranker_error = self._apply_reranker(question, candidates, top_k=requested_top_k)
        else:
            candidates = candidates[:requested_top_k]

        records: list[RetrievalRecord] = []
        for candidate in candidates[:requested_top_k]:
            payload = dict(candidate.get("metadata") or {})
            payload.update(
                {
                    "retrieval_strategy": requested_strategy,
                    "dense_score": candidate.get("dense_score"),
                    "lexical_score": candidate.get("lexical_score"),
                    "hybrid_seed_score": candidate.get("hybrid_seed_score"),
                    "reranker_score": candidate.get("reranker_score"),
                    "retrieval_sources": candidate.get("sources", []),
                }
            )
            if reranker_error:
                payload["reranker_error"] = reranker_error

            final_score = candidate.get("reranker_score")
            if final_score is None:
                final_score = candidate.get("hybrid_seed_score")
            if final_score is None:
                final_score = candidate.get("dense_score")
            if final_score is None:
                final_score = candidate.get("lexical_score")

            records.append(
                RetrievalRecord(
                    meeting_id=str(payload.get("meeting_id", "")),
                    document=str(candidate.get("document", "")),
                    score=None if final_score is None else round(float(final_score), 6),
                    metadata=payload,
                )
            )
        return records
