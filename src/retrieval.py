
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import chromadb


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "incident_knowledge"


class OllamaEmbeddingError(RuntimeError):
    """Raised when Ollama cannot generate embeddings."""


class IncidentRetriever:
    """Index incident knowledge and retrieve relevant documents locally."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.embedding_model = embedding_model

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        """Generate a text embedding using the local Ollama service."""

        start_time = time.perf_counter()

        payload = json.dumps({
            "model": self.embedding_model,
            "prompt": text,
        }).encode("utf-8")

        request = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as exc:
            logger.exception(
                "Embedding request failed | HTTP error | model=%s",
                self.embedding_model,
            )
            raise OllamaEmbeddingError(
                f"Ollama embedding request failed: HTTP {exc.code}."
            ) from exc

        except urllib.error.URLError as exc:
            logger.exception(
                "Embedding request failed | connection error | model=%s",
                self.embedding_model,
            )
            raise OllamaEmbeddingError(
                "Cannot connect to Ollama. Make sure Ollama is running."
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.exception(
                "Embedding response parsing failed | model=%s",
                self.embedding_model,
            )
            raise OllamaEmbeddingError(
                "Ollama returned an invalid embedding response."
            ) from exc

        embedding = (
            result.get("embedding")
            if isinstance(result, dict)
            else None
        )

        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in embedding
            )
        ):
            logger.error(
                "Invalid embedding returned by Ollama | model=%s",
                self.embedding_model,
            )
            raise OllamaEmbeddingError(
                "Ollama returned an invalid embedding."
            )

        duration = time.perf_counter() - start_time

        logger.info(
            "Embedding generated | model=%s | "
            "dimensions=%d | duration_seconds=%.3f",
            self.embedding_model,
            len(embedding),
            duration,
        )

        return embedding

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """Add or update documents in the ChromaDB collection."""

        if not documents:
            return 0

        ids = []
        texts = []
        metadatas = []
        embeddings = []

        seen_ids = set()

        for document in documents:
            if not isinstance(document, dict):
                raise ValueError(
                    "Each document must be a dictionary."
                )

            doc_id = document.get("id")
            text = document.get("text")

            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(
                    "Document ID must be a non-empty string."
                )

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Document text must be a non-empty string."
                )

            doc_id = doc_id.strip()
            text = text.strip()

            if doc_id in seen_ids:
                raise ValueError(
                    f"Duplicate document ID: {doc_id}"
                )

            seen_ids.add(doc_id)

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Metadata must be a dictionary."
                )

            for key, value in metadata.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        "Metadata keys must be non-empty strings."
                    )

                if not isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    raise ValueError(
                        "Metadata values must be strings, numbers, or booleans."
                    )

            safe_metadata = (
                metadata
                if metadata
                else {"source": "incident_knowledge"}
            )

            ids.append(doc_id)
            texts.append(text)
            metadatas.append(safe_metadata)

        # Generate all embeddings before modifying the collection.
        for text in texts:
            embeddings.append(self._embed(text))

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents, optionally filtering by cosine distance."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if max_distance is not None:
            if (
                not isinstance(max_distance, (int, float))
                or isinstance(max_distance, bool)
                or not 0 <= max_distance <= 2
            ):
                raise ValueError(
                    "max_distance must be between 0 and 2."
                )

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        embedding = self._embed(query)

        start_time = time.perf_counter()

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, collection_count),
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "Vector search completed | collection=%s | "
                "requested_results=%d | retrieved_results=%d | "
                "duration_seconds=%.3f",
                self.collection.name,
                top_k,
                len(results["ids"][0]),
                duration,
            )

        except Exception:
            logger.exception(
                "Vector search failed | collection=%s | requested_results=%d",
                self.collection.name,
                top_k,
            )
            raise

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        retrieved = []

        for index, distance in enumerate(distances):
            if (
                max_distance is not None
                and distance > max_distance
            ):
                continue

            retrieved.append({
                "id": ids[index],
                "text": documents[index],
                "metadata": metadatas[index] or {},
                "distance": distance,
            })

        return retrieved
