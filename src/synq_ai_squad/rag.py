"""Load the knowledge base into a searchable index (the vector store).

Run once, and again whenever you change files in knowledge/:
    uv run python -m synq_ai_squad.rag

What happens:
  1. LOAD   - read every .md file in knowledge/
  2. SPLIT  - cut each file into small chunks (one per "## heading" section)
  3. EMBED  - turn each chunk into a list of numbers that captures its meaning
  4. STORE  - save chunks + numbers in a small file (vector_store.json)
Later, a question gets turned into numbers too, and the store returns the chunks
whose numbers are closest, i.e. the chunks with the most similar meaning.

We use LangChain's simple in-memory store: for a few dozen chunks it is instant, and it's
tiny to deploy. (A dedicated vector database like Chroma or pgvector pays off at thousands
of chunks; we used Chroma at first, but it made the Render build too big for the free plan.)
"""

import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from synq_ai_squad.config import KNOWLEDGE_DIR, STORE_FILE, Settings, configure_logging, get_settings

log = logging.getLogger(__name__)


def embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, api_key=settings.require("google_api_key"))


def get_vectorstore(settings: Settings | None = None) -> InMemoryVectorStore:
    """Load the saved index from vector_store.json (run ingest() first to create it)."""
    if not STORE_FILE.exists():
        raise FileNotFoundError("No vector_store.json yet. Run: uv run python -m synq_ai_squad.rag")
    return InMemoryVectorStore.load(str(STORE_FILE), embeddings(settings or get_settings()))


def ensure_vectorstore(settings: Settings | None = None) -> InMemoryVectorStore:
    """Load the index, building it first if it's missing (e.g. the first start of a Docker container)."""
    if not STORE_FILE.exists():
        ingest(settings)
    return get_vectorstore(settings)


def load_and_split() -> list[Document]:
    # Split on headings first, so each chunk is one complete topic (e.g. one FAQ answer).
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=False,  # keep the heading text inside the chunk; it helps search
    )
    # Safety net: if a section is very long, cut it further into ~800-character pieces.
    by_size = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        for chunk in by_size.split_documents(by_heading.split_text(path.read_text(encoding="utf-8"))):
            chunk.metadata["source"] = path.name  # remember which file it came from
            # "Contextual chunk header": give every section its document title, so a chunk
            # like "## What does it cost?" also says it's about Synq Logic.
            if "section" in chunk.metadata:
                chunk.page_content = f"# {chunk.metadata['title']}\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def ingest(settings: Settings | None = None) -> None:
    chunks = load_and_split()
    store = InMemoryVectorStore(embeddings(settings or get_settings()))  # a fresh store, so no duplicates
    store.add_documents(chunks)
    store.dump(str(STORE_FILE))
    log.info(
        "Stored %d chunks from %d files in %s", len(chunks), len(list(KNOWLEDGE_DIR.glob("*.md"))), STORE_FILE.name
    )
    for c in chunks:
        log.info("   - %-22s | %s", c.metadata["source"], c.metadata.get("section", c.metadata.get("title", "")))


if __name__ == "__main__":
    configure_logging("%(message)s")
    ingest()
