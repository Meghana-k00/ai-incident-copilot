
from pathlib import Path

from src.retrieval import IncidentRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "docs"


def load_documents() -> list[dict]:
    """Load Markdown and text troubleshooting documents."""
    documents = []

    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".md", ".txt"
        }:
            continue

        content = path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append({
            "id": path.relative_to(KNOWLEDGE_DIR).as_posix(),
            "text": content,
            "metadata": {
                "source": path.name,
                "file_type": path.suffix.lower().lstrip("."),
            },
        })

    return documents


def main() -> None:
    if not KNOWLEDGE_DIR.exists():
        raise SystemExit(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    documents = load_documents()

    if not documents:
        raise SystemExit(
            f"No .md or .txt files found in {KNOWLEDGE_DIR}. "
            "Add troubleshooting documents before indexing."
        )

    retriever = IncidentRetriever()
    count = retriever.add_documents(documents)

    print(f"Indexed {count} knowledge documents.")
    print(f"ChromaDB collection: {retriever.collection.name}")


if __name__ == "__main__":
    main()
