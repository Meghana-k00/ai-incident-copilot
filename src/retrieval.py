
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import chromadb


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

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> list[float]:
        """Generate a text embedding using the local Ollama service."""
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
                request, timeout=120
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            raise OllamaEmbeddingError(
                f"Ollama embedding request failed: HTTP {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaEmbeddingError(
                "Cannot connect to Ollama. Make sure it is running."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaEmbeddingError(
                "Ollama returned an invalid embedding response."
            ) from exc

        embedding = result.get("embedding") if isinstance(
            result, dict
        ) else None

        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(
                isinstance(value, (int, float))
                for value in embedding
            )
        ):
            raise OllamaEmbeddingError(
                "Ollama returned an invalid embedding."
            )

        return embedding
    def add_documents(self, documents: list[dict[str, Any]]) -> int:
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
                raise ValueError("Each document must be a dictionary.")

            doc_id = document.get("id")
            text = document.get("text")

            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Document ID must be a non-empty string.")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Document text must be a non-empty string.")

            doc_id = doc_id.strip()
            text = text.strip()

            if doc_id in seen_ids:
                raise ValueError(f"Duplicate document ID: {doc_id}")

            seen_ids.add(doc_id)

            metadata = document.get("metadata", {})

            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary.")

            for key, value in metadata.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("Metadata keys must be non-empty strings.")

                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("Metadata values must be strings, numbers, or booleans.")

            # ChromaDB requires non-empty metadata dictionaries.
            safe_metadata = (
                metadata if metadata else {"source": "incident_knowledge"}
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
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant indexed documents."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a positive integer.")

        if top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if self.collection.count() == 0:
            return []

        embedding = self._embed(query)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        return [
            {
                "id": ids[index],
                "text": documents[index],
                "metadata": metadatas[index] or {},
                "distance": distances[index],
            }
            for index in range(len(ids))
        ]
