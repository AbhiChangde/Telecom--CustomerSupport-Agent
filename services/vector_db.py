import os
import asyncio
from datetime import datetime
from functools import lru_cache

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

# Cosine distance below this threshold = same/similar issue
# all-MiniLM cosine distances: 0.0 = identical, 1.0 = completely unrelated
# 0.40 catches paraphrases and same-category issues reliably
_SIMILARITY_THRESHOLD = 0.40


@lru_cache(maxsize=1)
def _get_collection():
    """
    Lazy-load ChromaDB + all-MiniLM-L6-v2 on first call.
    ~80 MB model download, cached after first run.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    print("\n[VectorDB] Loading ChromaDB + all-MiniLM-L6-v2 embedding model...", flush=True)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="customer_issues",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print("[VectorDB] Ready.\n", flush=True)
    return collection


# ── Synchronous internals (run in thread pool) ────────────────────────────────

def _sync_get_history(customer_id: str) -> list[dict]:
    """Return all stored issues for this customer, newest first."""
    collection = _get_collection()
    result = collection.get(
        where={"customer_id": customer_id},
        include=["metadatas"],
    )
    if not result["ids"]:
        return []
    issues = [
        {"intent": m.get("intent", ""), "date": m.get("session_date", "")}
        for m in result["metadatas"]
    ]
    issues.sort(key=lambda x: x["date"], reverse=True)
    return issues


def _sync_recall_similar(customer_id: str, issue_text: str) -> list[dict]:
    """Return past issues semantically similar to issue_text for this customer."""
    collection = _get_collection()
    existing = collection.get(where={"customer_id": customer_id})
    if not existing["ids"]:
        return []

    n = min(5, len(existing["ids"]))
    results = collection.query(
        query_texts=[issue_text],
        n_results=n,
        where={"customer_id": customer_id},
        include=["metadatas", "distances"],
    )

    similar = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if dist < _SIMILARITY_THRESHOLD:
            similar.append({
                "intent": meta.get("intent", ""),
                "date":   meta.get("session_date", ""),
                "similarity": round(1.0 - dist, 3),
            })
    return similar


def _sync_store(customer_id: str, intent: str, session_id: str) -> None:
    collection = _get_collection()
    collection.upsert(
        ids=[session_id],
        documents=[intent],
        metadatas=[{
            "customer_id":  customer_id,
            "intent":       intent,
            "session_date": datetime.now().strftime("%Y-%m-%d"),
        }],
    )
    print(f"[VectorDB] Stored: customer={customer_id}  issue=\"{intent}\"", flush=True)


# ── Public async API ──────────────────────────────────────────────────────────

async def recall_customer_history(customer_id: str) -> list[dict]:
    """
    All past issues logged for this customer, sorted newest first.
    Used at session start to inject memory into the system prompt.
    """
    if os.getenv("USE_MOCK_VECTOR_DB", "true").lower() == "true":
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_get_history, customer_id)


async def recall_similar_complaints(customer_id: str, issue_text: str) -> list[dict]:
    """
    Semantically similar past issues for this customer.
    Called in _close() once the current intent is known.
    Returns list of {intent, date, similarity} — empty list if none found.
    """
    if os.getenv("USE_MOCK_VECTOR_DB", "true").lower() == "true":
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_recall_similar, customer_id, issue_text)


async def store_conversation(customer_id: str, intent: str, session_id: str) -> None:
    """
    Persist the resolved issue so future sessions can recall it.
    Called from _close() on raise_ticket and escalate_human branches.
    """
    if os.getenv("USE_MOCK_VECTOR_DB", "true").lower() == "true":
        print(f"[MOCK VectorDB] Would store: customer={customer_id}  issue=\"{intent}\"")
        return
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_store, customer_id, intent, session_id)
