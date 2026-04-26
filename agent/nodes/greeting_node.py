from agent.state import SupportAgentState

GREETINGS = {
    "hi": "नमस्ते! Airtel customer support में आपका स्वागत है। आप अपनी समस्या बताइए।",
    "en": "Hello! Welcome to Airtel customer support. How can I help you today?",
    "ta": "வணக்கம்! Airtel வாடிக்கையாளர் சேவைக்கு வரவேற்கிறோம். உங்கள் பிரச்சனை என்ன?",
    "te": "నమస్కారం! Airtel కస్టమర్ సపోర్ట్‌కు స్వాగతం. మీ సమస్య ఏమిటి?",
    "mr": "नमस्कार! Airtel ग्राहक सेवेत आपले स्वागत आहे. आपली समस्या सांगा.",
    "bn": "নমস্কার! Airtel গ্রাহক সেবায় আপনাকে স্বাগতম। আপনার সমস্যা কী?",
}


async def greeting_node(state: SupportAgentState) -> SupportAgentState:
    # Greeting is surfaced via API response (greeting_url); no state changes needed.
    # Pre-cached audio is served from /static/greetings/{language_code}.mp3
    return state
