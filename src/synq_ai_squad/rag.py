"""Step 2a: load your documents into a searchable "memory" (the vector database).

Run once, and again whenever you change files in docs/:
    uv run python -m synq_ai_squad.rag

What happens:
  1. LOAD   - read every .md file in docs/
  2. SPLIT  - cut each file into small chunks (one per "## heading" section)
  3. EMBED  - turn each chunk into a list of numbers that captures its meaning
  4. STORE  - save chunks + numbers in a small file (vector_store.json)
Later, a question gets turned into numbers too, and the store returns the chunks
whose numbers are closest, i.e. the chunks with the most similar meaning.

We use LangChain's simple in-memory store: for a few dozen chunks it is instant, and it's
tiny to deploy. (A dedicated vector database like Chroma or pgvector pays off at thousands
of chunks; we used Chroma at first, but it made the Render build too big for the free plan.)
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
STORE_FILE = PROJECT_ROOT / "vector_store.json"


def embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def get_vectorstore() -> InMemoryVectorStore:
    """Load the saved store from vector_store.json (run this module first to create it)."""
    if not STORE_FILE.exists():
        raise FileNotFoundError("No vector_store.json yet. Run: uv run python -m synq_ai_squad.rag")
    return InMemoryVectorStore.load(str(STORE_FILE), embeddings())


def load_and_split() -> list[Document]:
    # Split on headings first, so each chunk is one complete topic (e.g. one FAQ answer).
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=False,  # keep the heading text inside the chunk; it helps search
    )
    # Safety net: if a section is very long, cut it further into ~800-character pieces.
    by_size = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for chunk in by_size.split_documents(by_heading.split_text(path.read_text(encoding="utf-8"))):
            chunk.metadata["source"] = path.name  # remember which file it came from
            # "Contextual chunk header": give every section its document title, so a chunk
            # like "## What does it cost?" also says it's about Synq Logic.
            if "section" in chunk.metadata:
                chunk.page_content = f"# {chunk.metadata['title']}\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def ingest() -> None:
    chunks = load_and_split()
    store = InMemoryVectorStore(embeddings())  # a fresh store every time, so no duplicates
    store.add_documents(chunks)
    store.dump(str(STORE_FILE))
    print(f"Stored {len(chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} files in {STORE_FILE.name}")
    for c in chunks:
        print(f"   - {c.metadata['source']:22} | {c.metadata.get('section', c.metadata.get('title', ''))}")


if __name__ == "__main__":
    ingest()
