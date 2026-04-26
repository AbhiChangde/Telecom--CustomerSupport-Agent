import os

# Known mock outages for demo scenarios
MOCK_OUTAGES = {
    "Hyderabad": "Partial 5G outage in Hyderabad West reported — ETA resolution 6 hours",
}


async def check_outage(location: str) -> str:
    if os.getenv("USE_MOCK_OUTAGE", "true").lower() == "true":
        return MOCK_OUTAGES.get(location, "none")

    raise NotImplementedError("Real outage API not implemented")
