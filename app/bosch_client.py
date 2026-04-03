import requests
from typing import Any

from app.config import (
    SUBSCRIPTION_KEY,
    BOSCH_URL,
    PROXIES,
    MODEL_MAX_TOKENS,
    MODEL_TEMPERATURE,
)


def ask_bosch(messages: list[dict[str, Any]]) -> str:
    headers = {
        "Content-Type": "application/json",
        "genaiplatform-farm-subscription-key": SUBSCRIPTION_KEY,
    }

    payload = {
        "messages": messages,
        "max_tokens": MODEL_MAX_TOKENS,
        "temperature": MODEL_TEMPERATURE,
        "stream": False,
    }

    response = requests.post(
        BOSCH_URL,
        headers=headers,
        json=payload,
        proxies=PROXIES,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    return str(content)