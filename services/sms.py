import os


async def send_sms(mobile: str, message: str) -> None:
    if os.getenv("USE_MOCK_SMS", "true").lower() == "true":
        print(f"[MOCK SMS] To {mobile}: {message}")
        return

    raise NotImplementedError("Real SMS integration not implemented")
