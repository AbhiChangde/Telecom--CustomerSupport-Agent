import os

MOCK_MEMORY = [
    {
        "complaint": "5G signal shows full bars but internet doesn't work",
        "resolution": "Network configuration reset required",
        "similarity": 0.91,
    },
    {
        "complaint": "Data balance depleting overnight without usage",
        "resolution": "Background app data identified, escalated to billing",
        "similarity": 0.85,
    },
]


async def recall_similar_complaints(customer_id: str, user_input: str) -> list[dict]:
    if os.getenv("USE_MOCK_VECTOR_DB", "true").lower() == "true":
        # Return mock similar complaints; in production this would be a semantic search
        return MOCK_MEMORY if any(
            kw in user_input.lower()
            for kw in ["5g", "data", "network", "signal", "internet"]
        ) else []

    provider = os.getenv("VECTOR_DB_PROVIDER", "pinecone")
    if provider == "pinecone":
        return await _pinecone_recall(customer_id, user_input)
    raise NotImplementedError(f"Vector DB provider '{provider}' not implemented")


async def store_conversation(customer_id: str, user_input: str, intent: str) -> None:
    if os.getenv("USE_MOCK_VECTOR_DB", "true").lower() == "true":
        print(f"[MOCK VECTOR DB] Stored conversation for {customer_id}: {intent}")
        return

    provider = os.getenv("VECTOR_DB_PROVIDER", "pinecone")
    if provider == "pinecone":
        await _pinecone_store(customer_id, user_input, intent)
        return
    raise NotImplementedError(f"Vector DB provider '{provider}' not implemented")


async def _pinecone_recall(customer_id: str, user_input: str) -> list[dict]:
    raise NotImplementedError("Pinecone integration not implemented")


async def _pinecone_store(customer_id: str, user_input: str, intent: str) -> None:
    raise NotImplementedError("Pinecone integration not implemented")
