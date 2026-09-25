"""Step 2a: load your documents into a searchable "memory" (the vector database).

Run once, and again whenever you change files in docs/:
    uv run python -m synq_ai_squad.rag

What happens:
  1. LOAD   - read every .md file in docs/
  2. SPLIT  - cut each file into small chunks (one per "## heading" section)
  3. EMBED  - turn each chunk into a list of numbers that captures its meaning
  4. STORE  - save chunks + numbers in Chroma, a database on your own computer
Later, a question gets turned into numbers too, and Chroma returns the chunks
whose numbers are closest, i.e. the chunks with the most similar meaning.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = PROJECT_ROOT / "chroma_db"


def get_vectorstore() -> Chroma:
    """Open the Chroma database (creates an empty one if it doesn't exist yet)."""
    return Chroma(
        collection_name="synq_docs",
        embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
        persist_directory=str(DB_DIR),
    )


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
    store = get_vectorstore()
    store.reset_collection()  # start fresh so re-running never creates duplicates
    store.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} files in {DB_DIR.name}/")
    for c in chunks:
        print(f"   - {c.metadata['source']:22} | {c.metadata.get('section', c.metadata.get('title', ''))}")


if __name__ == "__main__":
    ingest()
