
from unittest.mock import patch

import pytest

from src.retrieval import (
    IncidentRetriever,
    OllamaEmbeddingError,
)


@pytest.fixture
def retriever(tmp_path):
    """Create an isolated ChromaDB collection for each test."""
    return IncidentRetriever(
        persist_directory=tmp_path / "chroma_test",
        collection_name="test_incident_knowledge",
    )


def fake_embedding(text):
    """Return deterministic embeddings without requiring Ollama."""
    if "database" in text.lower():
        return [1.0, 0.0, 0.0]
    if "network" in text.lower():
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def test_add_documents_success(retriever):
    documents = [
        {
            "id": "incident-1",
            "text": "Database connection failures",
            "metadata": {"service": "database", "severity": "high"},
        },
        {
            "id": "incident-2",
            "text": "Network timeout failures",
            "metadata": {"service": "network", "severity": "medium"},
        },
    ]

    with patch.object(retriever, "_embed", side_effect=fake_embedding):
        added = retriever.add_documents(documents)

    assert added == 2
    assert retriever.collection.count() == 2


def test_add_empty_documents_returns_zero(retriever):
    assert retriever.add_documents([]) == 0
    assert retriever.collection.count() == 0


@pytest.mark.parametrize(
    "document",
    [
        {"id": "", "text": "Some incident"},
        {"id": 123, "text": "Some incident"},
        {"id": "incident-1", "text": ""},
        {"id": "incident-1", "text": "   "},
    ],
)
def test_add_documents_rejects_invalid_id_or_text(retriever, document):
    with pytest.raises(ValueError):
        retriever.add_documents([document])

    assert retriever.collection.count() == 0


def test_add_documents_rejects_duplicate_ids(retriever):
    documents = [
        {"id": "incident-1", "text": "First incident"},
        {"id": "incident-1", "text": "Second incident"},
    ]

    with pytest.raises(ValueError, match="Duplicate document ID"):
        retriever.add_documents(documents)

    assert retriever.collection.count() == 0


def test_add_documents_rejects_invalid_metadata(retriever):
    documents = [
        {
            "id": "incident-1",
            "text": "Database incident",
            "metadata": {"details": ["unsupported", "list"]},
        }
    ]

    with pytest.raises(ValueError, match="Metadata values"):
        retriever.add_documents(documents)


def test_add_documents_updates_existing_id(retriever):
    with patch.object(retriever, "_embed", side_effect=fake_embedding):
        retriever.add_documents([
            {"id": "incident-1", "text": "Old database incident"}
        ])
        retriever.add_documents([
            {"id": "incident-1", "text": "Updated database incident"}
        ])

    assert retriever.collection.count() == 1

    stored = retriever.collection.get(
        ids=["incident-1"],
        include=["documents"],
    )
    assert stored["documents"] == ["Updated database incident"]


def test_search_returns_relevant_document(retriever):
    with patch.object(retriever, "_embed", side_effect=fake_embedding):
        retriever.add_documents([
            {
                "id": "database-incident",
                "text": "Database connection failure",
                "metadata": {"service": "database"},
            },
            {
                "id": "network-incident",
                "text": "Network timeout failure",
                "metadata": {"service": "network"},
            },
        ])

        results = retriever.search("database failure", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "database-incident"
    assert results[0]["text"] == "Database connection failure"
    assert results[0]["metadata"]["service"] == "database"
    assert isinstance(results[0]["distance"], float)


def test_search_empty_collection_returns_empty_list(retriever):
    with patch.object(retriever, "_embed") as mock_embed:
        assert retriever.search("database failure") == []

    mock_embed.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_search_rejects_invalid_query(retriever, query):
    with pytest.raises(ValueError, match="Search query"):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "3"])
def test_search_rejects_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("database failure", top_k=top_k)


def test_embedding_failure_does_not_add_documents(retriever):
    with patch.object(
        retriever,
        "_embed",
        side_effect=OllamaEmbeddingError("Ollama unavailable"),
    ):
        with pytest.raises(OllamaEmbeddingError):
            retriever.add_documents([
                {"id": "incident-1", "text": "Database failure"}
            ])

    assert retriever.collection.count() == 0


def test_search_embedding_failure_is_propagated(retriever):
    with patch.object(retriever, "_embed", side_effect=fake_embedding):
        retriever.add_documents([
            {"id": "incident-1", "text": "Database failure"}
        ])

    with patch.object(
        retriever,
        "_embed",
        side_effect=OllamaEmbeddingError("Ollama unavailable"),
    ):
        with pytest.raises(OllamaEmbeddingError):
            retriever.search("database failure")
