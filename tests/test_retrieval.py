from __future__ import annotations

from meeting_ai.retrieval import MeetingVectorStore
from meeting_ai.schemas import SummaryResult, TranscriptResult, TranscriptSegment


class FakeEmbedder:
    def encode_texts(self, texts, task="passage"):
        vectors = []
        for text in texts:
            launch = 1.0 if "launch" in text.lower() else 0.0
            budget = 1.0 if "budget" in text.lower() else 0.0
            vectors.append([launch, budget])
        return vectors


class FakeCollection:
    def __init__(self):
        self.items: list[dict[str, object]] = []

    def upsert(self, ids, documents, metadatas, embeddings):
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.items.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }
            )

    def _matches_where(self, item: dict[str, object], where) -> bool:
        if where is None:
            return True
        if "$and" in where:
            return all(self._matches_where(item, clause) for clause in where["$and"])
        metadata = item["metadata"]
        for key, condition in where.items():
            if isinstance(condition, dict) and "$eq" in condition:
                if metadata.get(key) != condition["$eq"]:
                    return False
            elif metadata.get(key) != condition:
                return False
        return True

    def query(self, query_embeddings, n_results, include, where=None):
        query_vector = query_embeddings[0]
        filtered = [item for item in self.items if self._matches_where(item, where)]
        ranked = sorted(
            filtered,
            key=lambda item: sum(a * b for a, b in zip(item["embedding"], query_vector)),
            reverse=True,
        )[:n_results]
        return {
            "documents": [[item["document"] for item in ranked]],
            "metadatas": [[item["metadata"] for item in ranked]],
            "distances": [[0.0 for _ in ranked]],
        }


def build_transcript() -> TranscriptResult:
    segment = TranscriptSegment(speaker="SPEAKER_00", text="Launch budget discussion.", start=0.0, end=1.0)
    return TranscriptResult(
        audio_path="demo.wav",
        language="en",
        asr_model="mock",
        diarization_backend="mock",
        segments=[segment],
        full_text="[SPEAKER_00] Launch budget discussion.",
        metadata={},
    )


def test_vector_store_adds_summary_document() -> None:
    collection = FakeCollection()
    store = MeetingVectorStore(collection=collection, embedder=FakeEmbedder())

    meeting_id = store.add_summary(
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=["Share budget"]),
        transcript=build_transcript(),
        meeting_id="meeting-1",
    )

    assert meeting_id == "meeting-1"
    assert len(collection.items) == 2
    assert collection.items[0]["metadata"]["chunk_type"] == "summary"
    assert "Topics:" in collection.items[0]["document"]
    assert collection.items[0]["metadata"]["language"] == "en"
    assert collection.items[1]["metadata"]["chunk_type"] == "transcript"
    assert collection.items[1]["metadata"]["chunk_index"] == 0


def test_vector_store_queries_ranked_results() -> None:
    collection = FakeCollection()
    store = MeetingVectorStore(collection=collection, embedder=FakeEmbedder())
    store.add_summary(
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=[]),
        transcript=build_transcript(),
        meeting_id="launch-meeting",
    )
    store.add_summary(
        summary=SummaryResult(topics=["Budget"], decisions=["Cut spend"], follow_ups=[]),
        meeting_id="budget-meeting",
    )

    results = store.query("What did we decide about the launch?", top_k=1)

    assert len(results) == 1
    assert results[0].meeting_id == "launch-meeting"


def test_vector_store_queries_can_filter_by_chunk_type_and_meeting_id() -> None:
    collection = FakeCollection()
    store = MeetingVectorStore(collection=collection, embedder=FakeEmbedder())
    store.add_summary(
        summary=SummaryResult(topics=["Launch"], decisions=["Ship"], follow_ups=[]),
        transcript=build_transcript(),
        meeting_id="launch-meeting",
    )
    store.add_summary(
        summary=SummaryResult(topics=["Budget"], decisions=["Cut spend"], follow_ups=[]),
        meeting_id="budget-meeting",
    )

    transcript_results = store.query(
        "What did we discuss about launch?",
        top_k=3,
        chunk_type="transcript",
        meeting_id="launch-meeting",
    )
    summary_results = store.query(
        "What did we decide about budget?",
        top_k=3,
        chunk_type="summary",
        meeting_id="budget-meeting",
    )

    assert len(transcript_results) == 1
    assert transcript_results[0].metadata["chunk_type"] == "transcript"
    assert transcript_results[0].meeting_id == "launch-meeting"
    assert len(summary_results) == 1
    assert summary_results[0].metadata["chunk_type"] == "summary"
    assert summary_results[0].meeting_id == "budget-meeting"
